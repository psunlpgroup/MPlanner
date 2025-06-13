"""Model management for different backbones and AI models."""

import os
import sys
from typing import Optional
from abc import ABC, abstractmethod

sys.path.append('/scratch1/xzl5514/projects/utils')

import torch
from PIL import Image
import google.generativeai as genai
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from openai import OpenAI

from config import PathConfig, PlanningConfig

class BackboneInterface(ABC):
    """Abstract interface for different model backbones"""
    
    @abstractmethod
    def get_text_response(self, prompt: str, seed: int, temperature: float) -> Optional[str]:
        """Get text response from the model"""
        pass
    
    @abstractmethod
    def get_image_response(self, prompt: str, image_path: str, seed: int, temperature: float) -> Optional[str]:
        """Get response from model with image input"""
        pass

class GPT4OBackbone(BackboneInterface):
    """GPT-4O backbone implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_text_response(self, prompt: str, seed: int, temperature: float) -> Optional[str]:
        """Get text response from GPT-4O"""
        try:
            from gpt4o_funcs import get_response
            response = get_response(self.model_name, 'text', prompt, seed, temperature)
            return response if response != -1 else None
        except Exception as e:
            print(f"Error getting GPT-4O text response: {e}")
            return None
    
    def get_image_response(self, prompt: str, image_path: str, seed: int, temperature: float) -> Optional[str]:
        """Get response from GPT-4O with image"""
        try:
            from gpt4o_funcs import get_response
            response = get_response(self.model_name, 'image', prompt, seed, temperature, img_path=image_path)
            return response if response != -1 else None
        except Exception as e:
            print(f"Error getting GPT-4O image response: {e}")
            return None

class GeminiBackbone(BackboneInterface):
    """Gemini backbone implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
    
    def get_text_response(self, prompt: str, seed: int, temperature: float) -> Optional[str]:
        """Get text response from Gemini"""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"Error getting Gemini text response: {e}")
            return None
    
    def get_image_response(self, prompt: str, image_path: str, seed: int, temperature: float) -> Optional[str]:
        """Get response from Gemini with image"""
        try:
            image = Image.open(image_path)
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048,
            )
            response = self.model.generate_content([prompt, image], generation_config=generation_config)
            return response.text
        except Exception as e:
            print(f"Error getting Gemini image response: {e}")
            return None

class MistralBackbone(BackboneInterface):
    """Mistral backbone implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # TODO: Implement Mistral API integration
        print("Warning: Mistral backbone not fully implemented yet")
    
    def get_text_response(self, prompt: str, seed: int, temperature: float) -> Optional[str]:
        """Get text response from Mistral"""
        # TODO: Implement Mistral text generation
        print("Mistral text response not implemented")
        return None
    
    def get_image_response(self, prompt: str, image_path: str, seed: int, temperature: float) -> Optional[str]:
        """Get response from Mistral with image"""
        # TODO: Implement Mistral image understanding
        print("Mistral image response not implemented")
        return None

class ModelManager:
    """Manages all AI models and pipelines"""
    
    def __init__(self, config: PlanningConfig, path_config: PathConfig):
        self.config = config
        self.path_config = path_config
        self.backbone = self._init_backbone()
        self._init_vision_models()
    
    def _init_backbone(self) -> BackboneInterface:
        """Initialize the appropriate backbone"""
        backbone_map = {
            'gpt4o': GPT4OBackbone,
            'gem': GeminiBackbone,
            'mistral': MistralBackbone
        }
        
        backbone_class = backbone_map[self.config.backbone]
        return backbone_class(self.config.model)
    
    def _init_vision_models(self):
        """Initialize vision-related models"""
        # Stable Diffusion Pipeline
        self.pipe_stable = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            cache_dir=self.path_config.cache_dir, 
            torch_dtype=torch.float16
        ).to("cuda")
        
        # InstructPix2Pix Pipeline
        self.pipe_ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", 
            cache_dir=self.path_config.cache_dir, 
            torch_dtype=torch.float16, 
            safety_checker=None
        ).to("cuda")
        self.pipe_ip2p.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe_ip2p.scheduler.config
        )
        
        # BLIP2 Model
        self.processor_blip = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            cache_dir=self.path_config.cache_dir
        )
        self.model_blip = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            cache_dir=self.path_config.cache_dir, 
            torch_dtype=torch.float16, 
            device_map={"": 0}
        ).to("cuda")
    
    def get_text_response(self, prompt: str) -> Optional[str]:
        """Get text response using the configured backbone"""
        return self.backbone.get_text_response(
            prompt, self.config.seed, self.config.temperature
        )
    
    def get_image_response(self, prompt: str, image_path: str) -> Optional[str]:
        """Get image response using the configured backbone"""
        return self.backbone.get_image_response(
            prompt, image_path, self.config.seed, self.config.temperature
        )
    
    def generate_image(self, prompt: str) -> Image.Image:
        """Generate image using Stable Diffusion"""
        return self.pipe_stable(prompt).images[0]
    
    def edit_image(self, input_path: str, edit_prompt: str) -> Image.Image:
        """Edit image using InstructPix2Pix"""
        try:
            from InstructPix2PixMain.edit_cli import ip2p_main
            return ip2p_main(input=input_path, edit=edit_prompt)
        except ImportError:
            # Fallback to direct pipeline usage if import fails
            input_image = Image.open(input_path)
            return self.pipe_ip2p(edit_prompt, image=input_image).images[0]
    
    def get_image_caption(self, img_path: str) -> Optional[str]:
        """Generate caption for image using BLIP2"""
        try:
            prompt_cap = "Question: what does the image describe? Answer:"
            inputs = self.processor_blip(
                Image.open(img_path), prompt_cap, return_tensors="pt"
            ).to("cuda", torch.float16)
            generated_ids = self.model_blip.generate(**inputs)
            generated_text = self.processor_blip.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            return generated_text
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None