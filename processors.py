"""Task processors for different planning modes."""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image

from config import ORDER_DICT
from models import ModelManager

class TaskProcessor(ABC):
    """Base class for task processing"""
    
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager
    
    def process_task(self, task: str, task_idx: int, save_dir: str) -> bool:
        """Process a single task. Returns True if successful, False otherwise."""
        task_path = f"{save_dir}/task_{task_idx}"
        os.makedirs(task_path, exist_ok=True)
        
        try:
            return self._process_task_impl(task, task_path)
        except Exception as e:
            print(f"Error processing task {task_idx}: {e}")
            return False
    
    @abstractmethod
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        """Implementation-specific task processing"""
        pass
    
    def _save_text_file(self, content: str, filepath: str):
        """Save text content to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _load_text_file(self, filepath: str) -> Optional[str]:
        """Load text content from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

class PassageProcessor(TaskProcessor):
    """Processes tasks using passage-based planning"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ').lower()
        
        # Generate initial plan
        prompt_plan = (
            f"Please generate a step-by-step plan to {trunc_task}.\n"
            f"The plan should be less than {self.models.config.max_steps} steps.\n"
            f"Each step should include 3-5 sentences.\n"
            f"Between every two steps, please insert a line with a single character `-`."
        )
        
        plan = self.models.get_text_response(prompt_plan)
        if not plan:
            return False
        
        # Process steps
        steps = [step.strip() for step in plan.split('\n-\n')]
        ori_plan = "\n\n".join(steps)
        
        # Save original plan
        self._save_text_file(ori_plan, os.path.join(task_path, 'ori_plan.txt'))
        
        # Generate images for each step
        for i, step in enumerate(steps):
            img = self.models.generate_image(step)
            img_path = os.path.join(task_path, f"step_{i+1}.png")
            img.save(img_path)
        
        return True

class StepProcessor(TaskProcessor):
    """Processes tasks using step-by-step planning"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ')
        steps = []
        
        for k in range(self.models.config.max_steps):
            # Generate step prompt
            if k == 0:
                prompt_step = (
                    f"Please generate the first step of the plan to {trunc_task} "
                    f"in the following format:\nStep idx: Step Title\nStep Descriptions."
                )
            else:
                history_steps = "\n\n".join(steps)
                prompt_step = (
                    f"Please generate the next step to {trunc_task} following the given history steps.\n"
                    f"If the plan is completed, please type 'It is already done'.\n\n"
                    f"History Steps:\n{history_steps}\n"
                )
            
            # Get step response
            step = self.models.get_text_response(prompt_step)
            if not step:
                break
            
            if 'it is already done' in step.lower():
                break
            
            # Clean and store step
            step = re.sub(r'\n+', '\n', step)
            steps.append(step.strip())
            
            # Generate image
            if k == 0:
                img = self.models.generate_image(step)
            else:
                previous_img_path = os.path.join(task_path, f"step_{k}.png")
                img = self.models.edit_image(previous_img_path, step)
            
            img_path = os.path.join(task_path, f"step_{k+1}.png")
            img.save(img_path)
        
        # Save plan
        self._save_text_file("\n\n".join(steps), os.path.join(task_path, 'ori_plan.txt'))
        return True

class PassageTipProcessor(TaskProcessor):
    """Processes tasks using passage-based planning with tip enhancement"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ').lower()
        
        # Generate initial plan
        prompt_plan = (
            f"Please generate a step-by-step plan to {trunc_task}.\n"
            f"The plan should be less than {self.models.config.max_steps} steps.\n"
            f"Each step should include 3-5 sentences.\n"
            f"Between every two steps, please insert a line with a single character `-`."
        )
        
        plan = self.models.get_text_response(prompt_plan)
        if not plan:
            return False
        
        # Process steps
        steps = [step.strip() for step in plan.split('\n-\n')]
        ori_plan = "\n\n".join(steps)
        self._save_text_file(ori_plan, os.path.join(task_path, 'ori_plan.txt'))
        
        # Generate tips and captions
        captions = []
        for i, step in enumerate(steps):
            # Generate tip
            prompt_tip = f"{step}\nWhat do I need to draw in the picture to describe the above text?"
            tip = self.models.get_text_response(prompt_tip)
            if not tip:
                continue
            
            # Generate image from tip
            img = self.models.generate_image(tip)
            img_path = os.path.join(task_path, f"step_{i+1}.png")
            img.save(img_path)
            
            # Generate caption
            caption = self.models.get_image_caption(img_path)
            if caption:
                captions.append(f"Caption {i+1}: {caption}")
        
        # Save captions
        self._save_text_file("\n\n".join(captions), os.path.join(task_path, 'captions.txt'))
        
        # Generate revised plan
        prompt_revise = (
            f"Textual Instruction:\nTask: {task}?\n{ori_plan}\n\n"
            f"Visualized Instruction:\n{chr(10).join(captions)}\n\n"
            f"Rewrite the textual instruction with the knowledge from visualized instruction pair-wisely."
        )
        rev_plan = self.models.get_text_response(prompt_revise)
        if rev_plan:
            self._save_text_file(rev_plan, os.path.join(task_path, 'rev_plan.txt'))
        
        return True

class StepTipProcessor(TaskProcessor):
    """Processes tasks using step-by-step planning with tip enhancement"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ')
        steps = []
        descriptions = []
        rev_steps = []
        
        for k in range(self.models.config.max_steps):
            # Generate step prompt
            if k == 0:
                prompt_step = (
                    f"Please generate the first step of the plan to {trunc_task} "
                    f"in the following format:\nStep idx: Step Title\nStep Descriptions."
                )
            else:
                prompt_step = (
                    f"Please generate the next step to {trunc_task} following the given history steps.\n"
                    f"If the plan is completed, please type 'It is already done'.\n\n"
                    f"History Steps:\n{steps}\n"
                )
            
            # Get step response
            step = self.models.get_text_response(prompt_step)
            if not step:
                break
            
            if 'it is already done' in step.lower():
                break
            
            # Clean and store step
            step = re.sub(r'\n+', '\n', step)
            steps.append(step.strip())
            
            # Generate tip
            prompt_tip = f"{step.strip()}\nWhat do I need to draw in the picture to describe the above text?"
            tip = self.models.get_text_response(prompt_tip)
            if not tip:
                break
            
            # Generate image from tip
            img = self.models.generate_image(tip)
            img_path = os.path.join(task_path, f"step_{k+1}.png")
            img.save(img_path)
            
            # Generate caption
            caption = self.models.get_image_caption(img_path)
            if caption:
                descriptions.append(caption)
            
            # Generate revised step
            prompt_revise = (
                f"Rewrite the given {ORDER_DICT[k+1]} step to {trunc_task} "
                f"with the knowledge from visualized instruction.\n\n"
                f"{ORDER_DICT[k+1]} Step:\n{step}\n\n"
                f"Visualized Instruction:\n{caption}\n\n"
            )
            rev_step = self.models.get_text_response(prompt_revise)
            if rev_step:
                rev_steps.append(rev_step.strip())
        
        # Save all files
        self._save_text_file("\n\n".join(steps), os.path.join(task_path, 'ori_plan.txt'))
        self._save_text_file("\n\n".join(descriptions), os.path.join(task_path, 'descriptions.txt'))
        self._save_text_file("\n\n".join(rev_steps), os.path.join(task_path, 'rev_plan.txt'))
        
        return True

class PassageWithDescriptionProcessor(TaskProcessor):
    """Processes tasks using passage-based planning with detailed descriptions"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ').lower()
        
        # Generate initial plan
        prompt_plan = (
            f"Please generate a step-by-step plan to {trunc_task}.\n"
            f"The plan should be less than {self.models.config.max_steps} steps.\n"
            f"Each step should include 3-5 sentences.\n"
            f"Between every two steps, please insert a line with a single character `-`."
        )
        
        plan = self.models.get_text_response(prompt_plan)
        if not plan:
            return False
        
        # Process steps
        steps = [step.strip() for step in plan.split('\n-\n')]
        ori_plan = "\n\n".join(steps)
        self._save_text_file(ori_plan, os.path.join(task_path, 'ori_plan.txt'))
        
        # Generate images and descriptions
        img_descriptions = []
        for i, step in enumerate(steps):
            # Generate image
            if i == 0:
                img = self.models.generate_image(step)
            else:
                previous_img_path = os.path.join(task_path, f"step_{i}.png")
                img = self.models.edit_image(previous_img_path, step)
            
            img_path = os.path.join(task_path, f"step_{i+1}.png")
            img.save(img_path)
            
            # Get detailed description
            prompt_des = "Please describe the picture in details."
            des = self.models.get_image_response(prompt_des, img_path)
            if des:
                img_descriptions.append(des)
        
        # Save descriptions
        self._save_text_file("\n\n".join(img_descriptions), os.path.join(task_path, 'descriptions.txt'))
        
        # Generate revised plan
        add_know = '\n'.join([f"Step {i+1}: {img_descriptions[i]}" for i in range(len(img_descriptions))])
        prompt_revise = (
            f"Please revise the original plan with the knowledge from the visualized instruction pair-wisely.\n"
            f"Do not change the number of steps in the original plan.\n"
            f"Each step should include be in the following format:\nStep idx: Step Title\nStep Descriptions\n\n"
            f"\n\nOriginal Plan:\n\n{steps}\n\nVisualized Instruction:\n\n{add_know}\n"
        )
        revised_plan = self.models.get_text_response(prompt_revise)
        if revised_plan:
            self._save_text_file(revised_plan, os.path.join(task_path, 'rev_plan.txt'))
        
        return True

class PureImageProcessor(TaskProcessor):
    """Processes tasks by generating only images"""
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        trunc_task = task.lstrip('How to ')
        
        # Load reference article to determine number of steps
        ref_article_path = os.path.join(self.models.path_config.ref_path, f"task_{task_path.split('_')[-1]}.txt")
        
        try:
            with open(ref_article_path, 'r') as f:
                ref = f.read()
            n_steps = len(ref.split('\n\n'))
            n_steps = min(n_steps, 20)
        except:
            n_steps = 5  # Default fallback
        
        # Generate images for each step
        for k in range(n_steps):
            prompt_step = f"A picture of the {ORDER_DICT.get(k+1, f'{k+1}th')} step to {trunc_task}."
            img = self.models.generate_image(prompt_step)
            img_path = os.path.join(task_path, f"step_{k+1}.png")
            img.save(img_path)
        
        return True

class ImageToTextProcessor(TaskProcessor):
    """Processes existing images to generate text descriptions"""
    
    def __init__(self, model_manager: ModelManager, source_dir: str):
        super().__init__(model_manager)
        self.source_dir = source_dir
    
    def _process_task_impl(self, task: str, task_path: str) -> bool:
        # Find source images
        source_task_path = task_path.replace(self.models.path_config.save_dir, self.source_dir)
        
        if not os.path.exists(source_task_path):
            print(f"Source path not found: {source_task_path}")
            return False
        
        # Get all image files
        all_files = os.listdir(source_task_path)
        img_files = [f for f in all_files if f.endswith('.png')]
        num_steps = len(img_files)
        
        descriptions = []
        for i in range(num_steps):
            img_path = os.path.join(source_task_path, f"step_{i+1}.png")
            if os.path.exists(img_path):
                prompt_des = "Please describe the picture in details."
                des = self.models.get_image_response(prompt_des, img_path)
                if des:
                    des = re.sub(r'\n+', '\n', des)
                    des = f'Step {i+1}:\n{des}'
                    descriptions.append(des.strip())
        
        # Save descriptions as plan
        if descriptions:
            self._save_text_file("\n\n".join(descriptions), os.path.join(task_path, 'ori_plan.txt'))
            return True
        
        return False