import math, time, os
import numpy as np
from scipy.signal import savgol_filter
from typing import Callable
import cv2
from PIL import Image, ImageDraw, ImageColor
from modules.ui import plaintext_to_html
import modules.shared as shared
#from modules.processing import Processed, StableDiffusionProcessing
from modules.paths_internal import script_path
from .helpers import (
    fix_env_Path_ffprobe,
    closest_upper_divisible_by_eight,
    load_model_from_setting,
    do_upscaleImg,value_to_bool, find_ffmpeg_binary
)
from .sd_helpers import renderImg2Img, renderTxt2Img
from .image import (
    multiply_alpha, 
    shrink_and_paste_on_blank, 
    open_image, 
    apply_alpha_mask, 
    draw_gradient_ellipse, 
    resize_and_crop_image, 
    crop_feather_ellipse, 
    crop_inner_image, 
    apply_lut, 
    read_lut)
from .video import write_video, add_audio_to_video, ContinuousVideoWriter
from .InfZoomConfig import InfZoomConfig

progress = {
    "current_step": 0,
    "total_steps": 0,
    "percentage": 0,
}

def update_progress(current_step, total_steps):
    global progress
    progress["current_step"] = current_step
    progress["total_steps"] = total_steps
    progress["percentage"] = (current_step / total_steps) * 100

class InfZoomer:
    def __init__(self, config: InfZoomConfig) -> None:
        self.C = config
        self.prompts = {}
        self.prompt_images = {}
        self.prompt_alpha_mask_images = {}
        self.prompt_image_is_keyframe = {}
        self.main_frames = []
        self.out_config = {}

        for x in self.C.prompts_array:
            try:
                key = int(x[0])
                value = str(x[1])
                file_loc = str(x[2])
                alpha_mask_loc = str(x[3])
                is_keyframe = bool(x[4])
                self.prompts[key] = value
                self.prompt_images[key] = file_loc
                self.prompt_alpha_mask_images[key] = alpha_mask_loc
                self.prompt_image_is_keyframe[key] = value_to_bool(is_keyframe)
            except ValueError:
                pass

        assert len(self.C.prompts_array) > 0, "prompts is empty"
        print(str(len(self.prompts)) + " prompts found")
        print(str(len([value for value in self.prompt_images.values() if value != ""])) + " prompt Images found")
        print(str(len([value for value in self.prompt_alpha_mask_images.values() if value != ""])) + " prompt Alpha Masks found")

        fix_env_Path_ffprobe()
        self.out_config = self.prepare_output_path()

        self.current_seed = self.C.seed

        # knowing the mask_height and desired outputsize find a compromise due to align 8 contraint of diffuser
        self.width = closest_upper_divisible_by_eight(self.C.outputsizeW)
        self.height = closest_upper_divisible_by_eight(self.C.outputsizeH)

        if self.width > self.height:
            self.mask_width  = self.C.outpaint_amount_px 
            self.mask_height = math.trunc(self.C.outpaint_amount_px * self.height/self.width)  
        else:
            self.mask_height  = self.C.outpaint_amount_px  
            self.mask_width  = math.trunc(self.C.outpaint_amount_px * self.width/self.height)  

        # here we leave slightly the desired ratio since if size+2*mask_size % 8 != 0
        # distribute "aligning pixels" to the mask size equally. 
        # only consider mask_size since image size is alread 8-aligned
        self.mask_width -= self.mask_width % 4
        self.mask_height -= self.mask_height % 4

        assert 0 == (2*self.mask_width+self.width) % 8
        assert 0 == (2*self.mask_height+self.height) % 8

        print (f"Adapted sizes for diffusers to: {self.width}x{self.height}+mask:{self.mask_width}x{self.mask_height}. New ratio: {(self.width+self.mask_width)/(self.height+self.mask_height)} ")

        self.num_interpol_frames = round(self.C.video_frame_rate * self.C.zoom_speed) - 1 # keyframe not to be interpolated

        if (self.C.outpaintStrategy == "Corners"):
            self.fnOutpaintMainFrames = self.outpaint_steps_cornerStrategy
            self.fnInterpolateFrames = self.interpolateFramesOuterZoom
        elif (self.C.outpaintStrategy == "Center"):
           self.fnOutpaintMainFrames = self.outpaint_steps_v8hid
           self.fnInterpolateFrames = self.interpolateFramesSmallCenter
        else:
            raise ValueError("Unsupported outpaint strategy in Infinite Zoom")

        self.outerZoom = True    # scale from overscan to target viewport

    # object properties, different from user input config
    out_config = {}
    prompts = {}
    main_frames:Image = []
    start_frames:Image = []
    end_frames:Image = []

    outerZoom: bool
    mask_width: int
    mask_height: int
    current_seed: int
    contVW: ContinuousVideoWriter
    fnOutpaintMainFrames: Callable
    fnInterpolateFrames: Callable

    def create_zoom(self):
        for i in range(self.C.batchcount):
            print(f"Batch {i+1}/{self.C.batchcount}")
            result = self.create_zoom_single()
        return result

    def create_zoom_single(self):

        self.main_frames.append(self.prepareInitImage())

        load_model_from_setting("infzoom_inpainting_model", self.C.progress, "Loading Model for inpainting/img2img: ")

        processed = self.fnOutpaintMainFrames()

        if self.C.lut_filename != "":
            try:
                #processed = apply_lut(processed, self.C.lut_filename)
                self.main_frames = [apply_lut(frame, self.C.lut_filename) for frame in self.main_frames]
            except Exception as e:
                print(f"Skip LUT: Error applying LUT {str(e)}.")

        #trim frames that are blended or luma wiped
        self.start_frames = self.main_frames[:2]
        self.end_frames = self.main_frames[(len(self.main_frames) - 2):]
        if (self.C.blend_mode != 0) and ((int(self.C.video_start_frame_dupe_amount) > 0) or (int(self.C.video_last_frame_dupe_amount) > 0)):
            #trim first and last frames only from main_frames, store 2 frames in each start_frames and end_frames for blending
            self.main_frames = self.main_frames[1:(len(self.main_frames) - 1)]
        print(f"Trimmed Blending Mode frames: start_frames:{len(self.start_frames)} end_frames:{len(self.end_frames)} main_frames:{len(self.main_frames)}")

        if (self.C.upscale_do): 
            self.doUpscaling()

        #if self.C.video_zoom_mode:
        #    self.main_frames = self.main_frames[::-1]
        #    self.start_frames_temp = self.start_frames[::-1]
        #    self.start_frames = self.end_frames[::-1]
        #    self.end_frames = self.start_frames_temp
        #    self.start_frames_temp = None
        # This was not needed due to ffmpeg -vf reverse

        if not self.outerZoom:
            self.contVW = ContinuousVideoWriter(
                self.out_config["video_filename"], 
                self.main_frames[0],
                self.main_frames[1],
                self.C.video_frame_rate,
                int(self.C.video_start_frame_dupe_amount), 
                self.C.video_ffmpeg_opts,
                self.C.blend_invert_do,
                self.C.blend_image,
                self.C.blend_mode,
                self.C.blend_gradient_size,
                hex_to_rgba(self.C.blend_color)
            )
        
        self.fnInterpolateFrames() # changes main_frame and writes to video

        if (self.C.audio_filename is not None) and (len(self.C.audio_filename) > 0 ):
            self.out_config["video_filename"] = add_audio_to_video(self.out_config["video_filename"], self.C.audio_filename, str.replace(self.out_config["video_filename"], ".mp4", "_audio.mp4"), self.C.audio_volume, find_ffmpeg_binary())

        print("Video saved in: " + os.path.join(script_path, self.out_config["video_filename"]))

        return (
            self.out_config["video_filename"],
            self.main_frames,
            self.C.seed,
            processed.js(),
            plaintext_to_html(processed.info),
            plaintext_to_html(""),            
        )

    def doUpscaling(self):
        for idx,mf in enumerate(self.main_frames):
            print (f"\033[KInfZoom: Upscaling mainframe: {idx}   \r",end="")
            self.main_frames[idx]=do_upscaleImg(mf, self.C.upscale_do, self.C.upscaler_name, self.C.upscale_by)

        self.mask_width = math.trunc(self.mask_width*self.C.upscale_by)
        self.mask_height = math.trunc(self.mask_height *self.C.upscale_by)

        if self.C.outpaintStrategy == "Corners":
            self.width  = self.main_frames[0].width-2*self.mask_width 
            self.height = self.main_frames[0].height-2*self.mask_height
        else:
            self.width  = self.main_frames[0].width
            self.height = self.main_frames[0].height

    def prepareInitImage(self) -> Image:
        if self.C.custom_init_image:
            current_image = Image.new(mode="RGBA", size=(self.width, self.height))
            current_image = current_image.convert("RGB")
            current_image = cv2_to_pil(cv2.resize(
                    pil_to_cv2(self.C.custom_init_image),
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA            
                )
            )
            self.save2Collect(current_image, f"init_custom.png")
        else:
            if self.prompt_images[min(k for k in self.prompt_images.keys() if k >= 0)] == "":
                load_model_from_setting("infzoom_txt2img_model", self.C.progress, "Loading Model for txt2img: ")
                processed, self.C.seed = self.renderFirstFrame()
                if len(processed.images) > 0:
                    current_image = processed.images[0]
                    self.save2Collect(current_image, f"init_txt2img.png")
            else:
                print("using image 0 as Initial keyframe")
                current_image = open_image(self.prompt_images[min(k for k in self.prompt_images.keys() if k >= 0)])
                current_image = cv2_to_pil(cv2.resize(
                        pil_to_cv2(current_image),
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA            
                    )
                )
                self.save2Collect(current_image, f"init_custom.png")

        return current_image

    def renderFirstFrame(self):
        pr = self.getInitialPrompt()

        return renderTxt2Img(
                f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                self.C.negative_prompt,
                self.C.sampler,
                self.C.num_inference_steps,
                self.C.guidance_scale,
                self.current_seed,
                self.width,
                self.height
        )

    def getInitialPrompt(self):
        return self.prompts[min(k for k in self.prompts.keys() if k >= 0)]
    
    def outpaint_steps_cornerStrategy(self):
        current_image = self.main_frames[-1]
        exit_img = None

        # just 30 radius to get inpaint connected between outer and innter motive
        masked_image = create_mask_with_circles(
            *current_image.size, 
            self.mask_width, self.mask_height, 
            overmask=self.C.overmask, 
            radius=min(self.mask_height,self.mask_height)*0.25
        )

        new_width= masked_image.width
        new_height=masked_image.height

        outpaint_steps=self.C.num_outpainting_steps
        for i in range(outpaint_steps):            
            print (f"Outpaint step: {str(i + 1)}/{str(outpaint_steps)} Seed: {str(self.current_seed)}")
            update_progress(i + 1, outpaint_steps)
            current_image = self.main_frames[-1]
            alpha_mask = self.getAlphaMask(*current_image.size, i + 1)

            #keyframes are not outpainted
            try:
                paste_previous_image = not self.prompt_image_is_keyframe[(i + 1)]
                print(f"paste_prev_image: {paste_previous_image} {i + 1}")
            except KeyError:
                print(f"Your Prompt List is missing key {i + 1}")
                break

            if self.C.custom_exit_image and ((i + 1) == outpaint_steps):
                current_image = cv2_to_pil(cv2.resize(
                    pil_to_cv2(self.C.custom_exit_image),
                    (self.C.width, self.C.height), 
                    interpolation=cv2.INTER_AREA
                    )
                )
                
                #if 0 == self.outerZoom:
                exit_img = current_image.convert("RGBA")

                self.save2Collect(current_image, self.out_config, f"exit_img.png")
                paste_previous_image = False
            else:
                if self.prompt_images[max(k for k in self.prompt_images.keys() if k <= (i + 1))] == "":
                    expanded_image = cv2_to_pil(
                        cv2.resize(pil_to_cv2(current_image),
                                 (new_width,new_height),
                                 interpolation=cv2.INTER_AREA
                        )
                    )

                    #expanded_image = Image.new("RGB",(new_width,new_height),"black")
                    expanded_image.paste(current_image, (self.mask_width,self.mask_height))
                    pr = self.prompts[max(k for k in self.prompts.keys() if k <= i)]
                
                    processed, self.C.seed = renderImg2Img(
                        f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                        self.C.negative_prompt,
                        self.C.sampler,
                        self.C.num_inference_steps,
                        self.C.guidance_scale,
                        -1, # try to avoid massive repeatings: self.current_seed,
                        new_width,  #outpaintsizeW
                        new_height,  #outpaintsizeH
                        expanded_image,
                        masked_image,
                        self.C.inpainting_denoising_strength,
                        self.C.inpainting_mask_blur,
                        self.C.inpainting_fill_mode,
                        False, # self.C.inpainting_full_res,
                        0 #self.C.inpainting_padding,
                    )
                    #
                
                    if len(processed.images) > 0:
                        expanded_image = processed.images[0]
                        
                        if (self.outerZoom and not paste_previous_image):
                            zoomed_img = cv2_to_pil(cv2.resize(
                                    expanded_image,
                                    (self.width,self.height),
                                    interpolation=cv2.INTER_AREA
                                )
                            )
                            self.main_frames.append(zoomed_img)
                            processed.images[0]=self.main_frames[-1]
                            self.save2Collect(processed.images[0], f"key_frame_{i}.png")
                else:
                    # use prerendered image, known as keyframe. Resize to target size
                    print(f"image {i + 1} is a keyframe: {not paste_previous_image}")
                    current_image = open_image(self.prompt_images[(i + 1)])
                    current_image = resize_and_crop_image(current_image, self.width, self.height)
                    # apply predefined or generated alpha mask to current image:
                    current_image = apply_alpha_mask(current_image, self.getAlphaMask(*current_image.size, i + 1))
                    expanded_image = current_image

                    # if keyframe is last frame, use it as exit image
                    if (not paste_previous_image) and ((i + 1) == outpaint_steps):
                        exit_img = current_image
                        print("using keyframe as exit image")
                    elif (not paste_previous_image):
                        self.main_frames.append(current_image)
                        self.save2Collect(current_image, f"key_frame_{i + 1}.png")

                if paste_previous_image and i > 0:                    
                    zoomed_img = cv2_to_pil(cv2.resize(
                        pil_to_cv2(expanded_image),
                        (self.width,self.height), 
                        interpolation=cv2.INTER_AREA
                        )
                    )                       
                    
                    #if self.prompt_image_is_keyframe[(i + 1)]: # if previous frame was keyframe, use it as base for next frame                    
                    #   self.main_frames[-1] = expanded_image # replace small image
                    #   self.save2Collect(expanded_image, f"outpaint_step_e{i}.png")
                        
                    if (i < outpaint_steps-1):
                        self.main_frames.append(zoomed_img)   # prepare next frame with former content
                        self.save2Collect(zoomed_img, f"outpaint_step_z{i + 1}.png")

            self.main_frames[-1] = apply_alpha_mask(self.main_frames[-1],  multiply_alpha(alpha_mask, 0.75))

        if exit_img is not None:
            self.main_frames.append(exit_img)

        return processed
   
    def outpaint_steps_v8hid(self):

        prev_image = self.main_frames[0].convert("RGBA")
        exit_img = self.C.custom_exit_image.convert("RGBA") if self.C.custom_exit_image else None

        outpaint_steps=self.C.num_outpainting_steps
        processed = [] # list of processed images, in the event there is nothing to actually process

        self.fixMaskSizes()

        for i in range(outpaint_steps):                
            print (f"Outpaint step: {str(i + 1)} / {str(outpaint_steps)} Seed: {str(self.current_seed)} \r")
            update_progress(i + 1, outpaint_steps)
            current_image = self.main_frames[-1]

            reduced_image = shrink_and_paste_on_blank(
                current_image.copy(), self.mask_width , self.mask_height
            )

            mask_image = np.array(reduced_image)[:, :, 3]
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")
            # create mask (black image with white mask_width width edges)

            #keyframes are not inpainted
            try:
                paste_previous_image = not self.prompt_image_is_keyframe[(i + 1)]
                print(f"paste_prev_image: {paste_previous_image} {i} {i + 1}")
            except KeyError:
                print(f"Your Prompt List is missing key {i + 1}")
                break

            if self.C.custom_exit_image and ((i + 1) == outpaint_steps):
                current_image = cv2_to_pil(
                    cv2.resize( pil_to_cv2(
                        self.C.custom_exit_image),
                        (self.width, self.height), 
                        interpolation=cv2.INTER_AREA)
                )                
                exit_img = current_image.convert("RGBA")
                # print("using Custom Exit Image")
                self.save2Collect(current_image, f"exit_img.png")

                paste_previous_image = False
            else:
                if self.prompt_images[max(k for k in self.prompt_images.keys() if k <= (i + 1))] == "":
                    pr = self.prompts[max(k for k in self.prompts.keys() if k <= i)]

                    processed, self.C.seed = renderImg2Img(
                        f"{self.C.common_prompt_pre}\n{pr}\n{self.C.common_prompt_suf}".strip(),
                        self.C.negative_prompt,
                        self.C.sampler,
                        self.C.num_inference_steps,
                        self.C.guidance_scale,
                        -1, #self.current_seed,
                        self.width,
                        self.height,
                        reduced_image,
                        mask_image,
                        self.C.inpainting_denoising_strength,
                        self.C.inpainting_mask_blur,
                        self.C.inpainting_fill_mode,
                        False, #self.C.inpainting_full_res,
                        0 #self.C.inpainting_padding,
                    )

                    if len(processed.images) > 0:
                        current_image = processed.images[0].convert("RGBA")
                        self.main_frames.append(current_image)
                        self.save2Collect(processed.images[0], f"outpain_step_{i}.png")
                else:
                     # use prerendered image, known as keyframe. Resize to target size
                    print(f"image {i + 1} is a keyframe: {not paste_previous_image}")
                    current_image = open_image(self.prompt_images[(i + 1)])
                    current_image = resize_and_crop_image(current_image, self.width, self.height).convert("RGBA")

                    # if keyframe is last frame, use it as exit image
                    if (not paste_previous_image) and ((i + 1) == outpaint_steps):
                        exit_img = current_image
                        print("using keyframe as exit image")
                    else:
                        # apply predefined or generated alpha mask to current image:
                        current_image = apply_alpha_mask(current_image, self.getAlphaMask(*current_image.size, i + 1))
                        self.main_frames.append(current_image)
                    self.save2Collect(current_image, f"key_frame_{i + 1}.png")

                # TODO: seed behavior

            # paste current image with alpha layer on previous image to merge : paste on i         
            if paste_previous_image and i > 0:
                # apply predefined or generated alpha mask to current image: 
                # current image must be redefined as most current image in frame stack
                # use previous image alpha mask if available
                current_image = apply_alpha_mask(self.main_frames[i + 1], self.getAlphaMask(*self.main_frames[i + 1].size, i + 1))

                #handle previous image alpha layer
                #prev_image = (main_frames[i] if main_frames[i] else main_frames[0])
                ## apply available alpha mask of previous image (inverted)
           
                prev_image_amask = self.getAlphaMask(self.width, self.height ,i, False)
                #prev_image = apply_alpha_mask(prev_image, prev_image_amask, invert = True)

                # merge previous image with current image
                corrected_frame = crop_inner_image(
                    current_image, self.mask_width, self.mask_height
                )
                prev = Image.new(prev_image.mode, (self.width, self.height), (255,255,255,255))
                prev.paste(apply_alpha_mask(self.main_frames[i], prev_image_amask))
                corrected_frame.paste(prev, mask=prev)
                
                self.main_frames[i] = corrected_frame
                self.save2Collect(corrected_frame, f"main_frame_gradient_{i + 0}")

        if exit_img is not None:
            self.main_frames.append(exit_img)
        return processed

    def calculate_interpolation_steps_linear(self, original_size, target_size, steps):
        width, height = original_size
        target_width, target_height = target_size

        if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0 or steps <= 0:
            return []

        width_step = (width - target_width) / (steps+1)     #+1 enforce steps BETWEEN keyframe, dont reach the target size. interval  like []
        height_step = (height - target_height) / (steps+1)

        scaling_steps = [(round(width - i * width_step), round(height - i * height_step)) for i in range(1,steps+1)]
        #scaling_steps.insert(0,original_size) # initial size is in the list
        return scaling_steps
       
    def interpolateFramesOuterZoom(self):
        blend_invert = self.C.blend_invert_do
        
        if self.C.video_zoom_mode:
            self.C.video_ffmpeg_opts += "-vf reverse"
            # note worked in video tab for ffmpeg expert mode: -vf lutyuv=u='(val-maxval/2)*2+maxval/2':v='(val-maxval/2)*2+maxval/2'
            # this may mean that if -vf is used in expert mode, we need to add reverse with a comma in the -vf argument
            blend_invert = not self.C.blend_invert_do   

        #if 0 == self.C.video_zoom_mode:
        current_image = self.start_frames[0]
        next_image = apply_alpha_mask(self.start_frames[1], self.getAlphaMask(*self.start_frames[1].size, 1))
        #elif 1 == self.C.video_zoom_mode:
        #    current_image = self.main_frames[-1]
        #    next_image = self.main_frames[-2]
        #else:
        #    raise ValueError("unsupported Zoom mode in INfZoom")

        outzoomSize = (self.width+self.mask_width*2, self.height+self.mask_height*2)
        target_size = (self.width, self.height) # mask border, hide blipping

        scaling_steps = self.calculate_interpolation_steps_linear(outzoomSize, target_size, self.num_interpol_frames)
        print(f"Before: {scaling_steps}, length: {len(scaling_steps)}")

        # all sizes EVEN
        for i,s in enumerate(scaling_steps):
            scaling_steps[i] = (s[0]+s[0]%2, s[1]+s[1]%2)
            # ODD steps producing jumps. even steps on even resolution is what we need.
            #scaling_steps[i] = (s[0]+1, s[1]+1)

        print(f"After EVEN: {scaling_steps}, length: {len(scaling_steps)}")


        def calculate_differences(lst):
            # Es wird eine leere Liste initialisiert, in der die Differenzen gespeichert werden
            diff_lst = []

            # Durchlaufen der Liste
            for i in range(1, len(lst)):
                # Differenz zwischen aufeinanderfolgenden Tupeln berechnen
                diff = (lst[i][0]-lst[i-1][0], lst[i][1]-lst[i-1][1])
                # Die Differenz zum diff_lst hinzuf�gen
                diff_lst.append(diff)

            return diff_lst

        # Beispielliste von Tupeln
        print(calculate_differences(scaling_steps))




        print ("Ratios:")
        for s in scaling_steps:
            print(f"{str(s[0]/s[1])}",end=";")

        self.contVW = ContinuousVideoWriter(self.out_config["video_filename"], 
                                            self.cropCenterTo(current_image.copy(),(target_size)),
                                            apply_alpha_mask(self.cropCenterTo(next_image.copy(),(target_size)),next_image.split()[3]),
                                            self.C.video_frame_rate,int(self.C.video_start_frame_dupe_amount-1),
                                            self.C.video_ffmpeg_opts,
                                            blend_invert,
                                            self.C.blend_image,
                                            self.C.blend_mode,
                                            self.C.blend_gradient_size,
                                            hex_to_rgba(self.C.blend_color))

        if  len(self.main_frames) <2:
            raise ValueError("Not enough frames for interpolation, possibly disable Luma Wipes")
        # Build interpolation frames
        # last frame skip interpolation
        for i in range(1, len(self.main_frames)):
            alpha_mask = self.getAlphaMask(*target_size, i)
            current_image = apply_alpha_mask(self.main_frames[0+i],alpha_mask)
            #previous_image = self.main_frames[i-1]

            lastFrame =  self.cropCenterTo(current_image.copy(),target_size)
            
            #self.contVW.append([current_image])
            #self.contVW.append([lastFrame])
            
            cv2_image = pil_to_cv2(current_image)

            # Resize and crop using OpenCV2
            for j in range(self.num_interpol_frames):
                print(f"\033[KInfZoom: Interpolate frame(CV2): main/inter: {i}/{j}   \r", end="")
                resized_image = cv2.resize(
                    cv2_image,
                    (scaling_steps[j][0], scaling_steps[j][1]),
                    interpolation=cv2.INTER_AREA
                )
                cropped_image_cv2 = cv2_crop_center(resized_image, target_size)
                cropped_image_pil = cv2_to_pil(cropped_image_cv2)
                
                self.contVW.append([cropped_image_pil])
                #lastFrame = cropped_image_pil
            #self.contVW.append([lastFrame])

        # process last main frame
        if (self.main_frames[-1].size != target_size):
            print(f"size error:        {self.main_frames[-1].size} != {target_size} {i}/{j}   \r", end="")
            self.main_frames[-1] = resize_and_crop_image(self.main_frames[-1],*target_size)
        self.contVW.append([self.main_frames[-1]])

        # process last frames
        lastFrame = self.end_frames[1]
        nextToLastFrame = self.end_frames[0]
            
        self.contVW.finish(lastFrame,
                        nextToLastFrame,
                        int(self.C.video_last_frame_dupe_amount),
                        self.C.blend_invert_do,
                        self.C.blend_image,
                        self.C.blend_mode,
                        self.C.blend_gradient_size,
                        hex_to_rgba(self.C.blend_color))

        """ USING PIL:
        for i in range(len(self.main_frames)):
            if 0 == self.C.video_zoom_mode:
                current_image = self.main_frames[0+i]
            else:
                current_image = self.main_frames[-1-i]

            self.contVW.append([
                self.cropCenterTo(current_image,(self.width, self.height))
            ])

            # interpolation steps between 2 inpainted images (=sequential zoom and crop)
            for j in range(self.num_interpol_frames - 1):
                print (f"\033[KInfZoom: Interpolate frame: main/inter: {i}/{j}   \r",end="")
                #todo: howto zoomIn when writing each frame; self.main_frames are inverted, howto interpolate?
                scaled_image = current_image.resize(scaling_steps[j], Image.LANCZOS)                    
                cropped_image = self.cropCenterTo(scaled_image,(self.width, self.height))

                self.contVW.append([cropped_image])
        """

    def interpolateFramesSmallCenter(self):
        blend_invert = self.C.blend_invert_do  
        
        if self.C.video_zoom_mode:
            self.C.video_ffmpeg_opts += "-vf reverse"
            # note worked in video tab for ffmpeg expert mode: -vf lutyuv=u='(val-maxval/2)*2+maxval/2':v='(val-maxval/2)*2+maxval/2'
            # this may mean that if -vf is used in expert mode, we need to add reverse with a comma in the -vf argument
            blend_invert = not self.C.blend_invert_do        

        self.contVW = ContinuousVideoWriter(self.out_config["video_filename"],
                                self.start_frames[0],#(self.width,self.height)),
                                self.start_frames[1],#(self.width,self.height)),
                                self.C.video_frame_rate,int(self.C.video_start_frame_dupe_amount),
                                self.C.video_ffmpeg_opts,
                                blend_invert,
                                self.C.blend_image,
                                self.C.blend_mode,
                                self.C.blend_gradient_size,
                                hex_to_rgba(self.C.blend_color))        

        for i in range(len(self.main_frames) - 1):
            # interpolation steps between 2 inpainted images (=sequential zoom and crop)
            for j in range(self.num_interpol_frames - 1):

                print (f"\033[KInfZoom: Interpolate frame: main/inter: {i}/{j}   \r",end="")
                #todo: howto zoomIn when writing each frame; self.main_frames are inverted, howto interpolate?
                current_image = self.main_frames[i + 1]


                interpol_image = current_image
                self.save2Collect(interpol_image, f"interpol_img_{i}_{j}].png")

                interpol_width, interpol_height, interpol_width2, interpol_height2 = self.getInterpol(j) # calculate interpolation values
                #print(f"\033[interpol_width, interpol_height, interpol_width2, interpol_height2: {interpol_width, interpol_height, interpol_width2, interpol_height2} \r")

                interpol_image = interpol_image.crop(
                    (
                        interpol_width,
                        interpol_height,
                        self.width - interpol_width,
                        self.height - interpol_height,
                    )
                )

                interpol_image = interpol_image.resize((self.width, self.height))
                self.save2Collect(interpol_image, f"interpol_resize_{i}_{j}.png")

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                prev_image_fix_crop = shrink_and_paste_on_blank(
                    self.main_frames[i], interpol_width2, interpol_height2
                )

                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)
                self.save2Collect(interpol_image, f"interpol_prevcrop_{i}_{j}.png")

                self.contVW.append([interpol_image])

            self.contVW.append([current_image])
        # process last frames
        lastFrame = self.end_frames[1]
        nextToLastFrame = self.end_frames[0]

        self.contVW.finish(lastFrame,
                        nextToLastFrame,
                        int(self.C.video_last_frame_dupe_amount),
                        blend_invert,
                        self.C.blend_image,
                        self.C.blend_mode,
                        self.C.blend_gradient_size,
                        hex_to_rgba(self.C.blend_color))

        #if self.C.video_zoom_mode:
        #    result = self.contVW.reverse_frames()
        #    print(f"reverse result: {result.stdout.decode}")


    def prepare_output_path(self):
        isCollect = shared.opts.data.get("infzoom_collectAllResources", False)
        output_path = shared.opts.data.get("infzoom_outpath", "outputs")

        save_path = os.path.join(
            output_path, shared.opts.data.get("infzoom_outSUBpath", "infinite-zooms")
        )

        if isCollect:
            save_path = os.path.join(save_path, "iz_collect" + str(int(time.time())))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_filename = os.path.join(
            save_path, "infinite_zoom_" + str(int(time.time())) + ".mp4"
        )

        return {
            "isCollect": isCollect,
            "save_path": save_path,
            "video_filename": video_filename,
        }


    def save2Collect(self, img, name):
        if self.out_config["isCollect"]:
            img.save(f'{self.out_config["save_path"]}/{name}.png')


    def frame2Collect(self,all_frames):
        self.save2Collect(all_frames[-1], self.out_config, f"frame_{len(all_frames)}")


    def frames2Collect(self, all_frames):
        for i, f in enumerate(all_frames):
            self.save2Collect(f, self.out_config, f"frame_{i}")

    def cropCenterTo(self, im: Image, toSize: tuple[int,int]):
        width, height = im.size
        left = (width - toSize[0])//2
        top = (height - toSize[1])//2
        right = (width + toSize[0])//2
        bottom = (height + toSize[1])//2
        return im.crop((left, top, right, bottom))

    def getAlphaMask(self, width, height, key, invert:bool = False):
        from PIL import ImageOps

        if self.prompt_alpha_mask_images[max(k for k in self.prompt_alpha_mask_images.keys() if k <= (key))] != "":
            image_alpha_mask = open_image(self.prompt_alpha_mask_images[max(k for k in self.prompt_alpha_mask_images.keys() if  k <= (key))])
        else:
            image_gradient_ratio = (self.C.blend_gradient_size / 100)
            image_alpha_mask = draw_gradient_ellipse(width, height, image_gradient_ratio, 0.0, 2.5)
        if invert:
            image_alpha_mask = ImageOps.invert(image_alpha_mask.convert('L'))
        return image_alpha_mask.convert('L')
    
    def getInterpol(self,j:int = 0):
        interpol_width = math.ceil(
        ( 1 - (1 - 2 * self.mask_width / self.width) **(1 - (j + 1) / self.num_interpol_frames) ) 
            * self.width / 2
        )

        interpol_height = math.ceil(
            ( 1 - (1 - 2 * self.mask_height / self.height) ** (1 - (j + 1) / self.num_interpol_frames) )
            * self.height/2
        )

        interpol_width2 = math.ceil(
            (1 - (self.width - 2 * self.mask_width) / (self.width - 2 * interpol_width))
            / 2 * self.width
        )

        interpol_height2 = math.ceil(
            (1 - (self.height - 2 * self.mask_height) / (self.height - 2 * interpol_height))
            / 2 * self.height
        )
        return interpol_width, interpol_height, interpol_width2, interpol_height2

    def fixMaskSizes(self):
        # This is overkill, as it clips the values twice, but it's the easiest way to ensure the values are correct
        mask_width = self.mask_width
        mask_height = self.mask_height
        # set minimum mask size to 12.5% of the image size
        if mask_width < self.width // 8:
            mask_width = self.width // 8
            mask_height = self.height // 8
            print(f"\033[93m{self.mask_width}x{self.mask_height} set - used: {mask_width}x{mask_height} Recommend: {self.width // 4}x{self.height // 4} Correct in Outpaint pixels settings.")
        # set maximum mask size to 75% of the image size
        if mask_width > (self.width // 4) * 3:
            mask_width = (self.width // 4) * 3
            mask_height = (self.height // 4) * 3
            print(f"\033[93m{self.mask_width}x{self.mask_height} set - used: {mask_width}x{mask_height} Recommend: {self.width // 4}x{self.height // 4} Correct in Outpaint pixels settings.")

        #self.mask_width = np.clip(int(mask_width), self.width // 8, (self.width // 4) * 3)
        #self.mask_height = np.clip(int(mask_height), self.width // 8, (self.width // 4) * 3)
##########################################################################################################################

##########################################################################################################################
# Infinite Zoom Classic

def prepare_output_path():
    isCollect = shared.opts.data.get("infzoom_collectAllResources", False)
    output_path = shared.opts.data.get("infzoom_outpath", "outputs")

    save_path = os.path.join(
        output_path, shared.opts.data.get("infzoom_outSUBpath", "infinite-zooms")
    )

    if isCollect:
        save_path = os.path.join(save_path, "iz_collect" + str(int(time.time())))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    video_filename = os.path.join(
        save_path, "infinite_zoom_" + str(int(time.time())) + ".mp4"
    )

    return {
        "isCollect": isCollect,
        "save_path": save_path,
        "video_filename": video_filename,
    }


def save2Collect(img, out_config, name):
    if out_config["isCollect"]:
        img.save(f'{out_config["save_path"]}/{name}.png')


def frame2Collect(all_frames, out_config):
    save2Collect(all_frames[-1], out_config, f"frame_{len(all_frames)}")


def frames2Collect(all_frames, out_config):
    for i, f in enumerate(all_frames):
        save2Collect(f, out_config, f"frame_{i}")

#################################################################################################################
def create_mask_with_circles(original_image_width, original_image_height, border_width, border_height, overmask: int, radius=4):
    # Create a new image with border and draw a mask
    new_width = original_image_width + 2 * border_width
    new_height = original_image_height + 2 * border_height

    # Create new image, default is black
    mask = Image.new('RGB', (new_width, new_height), 'white')

    # Draw black rectangle
    draw = ImageDraw.Draw(mask)
    draw.rectangle([border_width+overmask, border_height+overmask, new_width - border_width-overmask, new_height - border_height-overmask], fill='black')

    # Coordinates for circles
    circle_coords = [
        (border_width, border_height),  # Top-left
        (new_width - border_width, border_height),  # Top-right
        (border_width, new_height - border_height),  # Bottom-left
        (new_width - border_width, new_height - border_height),  # Bottom-right
        (new_width // 2, border_height),  # Middle-top
        (new_width // 2, new_height - border_height),  # Middle-bottom
        (border_width, new_height // 2),  # Middle-left
        (new_width - border_width, new_height // 2)  # Middle-right
    ]

    # Draw circles
    for coord in circle_coords:
        draw.ellipse([coord[0] - radius, coord[1] - radius, coord[0] + radius, coord[1] + radius], fill='white')
    return mask


def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cv2_crop_center(img, toSize: tuple[int,int]):
    y,x = img.shape[:2]
    startx = x//2-(toSize[0]//2)
    starty = y//2-(toSize[1]//2)
    return img[starty:starty+toSize[1],startx:startx+toSize[0]]

def hex_to_rgba(hex_color):
    try:
        # Convert hex color to RGBA tuple
        rgba = ImageColor.getcolor(hex_color, "RGBA")
    except ValueError:
        # If the hex color is invalid, default to yellow
        rgba = (255,255,0,255)
    return rgba