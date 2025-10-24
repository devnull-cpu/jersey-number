import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
from pathlib import Path
import random

class ImageLabeller:
    def __init__(self, image_dir, folder_labels_file, output_labels_file):
        self.image_dir = Path(image_dir)
        self.output_labels_file = output_labels_file
        
        # Load folder labels
        with open(folder_labels_file, 'r') as f:
            self.folder_labels = json.load(f)
        
        # Get all images from folders that aren't -1
        print("Loading images from valid folders...")
        self.images = []
        for folder_name, label in self.folder_labels.items():
            if label == -1:
                continue  # Skip -1 folders
            
            folder_path = self.image_dir / folder_name
            if folder_path.exists():
                for img_path in folder_path.rglob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append({
                            'path': str(img_path),
                            'folder': folder_name,
                            'folder_label': label
                        })
        
        print(f"Found {len(self.images):,} images from {len([l for l in self.folder_labels.values() if l != -1])} valid folders")
        
        # Shuffle for random order
        random.shuffle(self.images)
        
        # Load existing labels if they exist
        self.labels = {}
        if Path(output_labels_file).exists():
            with open(output_labels_file, 'r') as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} existing labels")
        
        self.current_idx = 0
        self.session_count = 0
        self.label_history = []  # Track labeling history for undo
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Jersey Number Visibility Labeller")
        self.root.geometry("1000x800")
        
        # Info panel
        self.info_frame = ttk.Frame(self.root, padding="10")
        self.info_frame.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(self.info_frame, text="", font=('Arial', 12))
        self.progress_label.pack()
        
        self.folder_label = ttk.Label(self.info_frame, text="", font=('Arial', 10))
        self.folder_label.pack()
        
        # Image display
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Button frame
        self.button_frame = ttk.Frame(self.root, padding="10")
        self.button_frame.pack(fill=tk.X)
        
        # Back/Undo button
        self.back_button = tk.Button(
            self.button_frame, 
            text="← BACK / UNDO (B)", 
            command=self.go_back,
            bg='#9E9E9E',
            fg='white',
            font=('Arial', 16, 'bold'),
            height=2
        )
        self.back_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        # Large buttons for Yes/No
        self.yes_button = tk.Button(
            self.button_frame, 
            text="✓ YES - Number Visible (Y)", 
            command=self.label_yes,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 16, 'bold'),
            height=2
        )
        self.yes_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        self.no_button = tk.Button(
            self.button_frame, 
            text="✗ NO - Number Not Visible (N)", 
            command=self.label_no,
            bg='#f44336',
            fg='white',
            font=('Arial', 16, 'bold'),
            height=2
        )
        self.no_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        self.skip_button = tk.Button(
            self.button_frame, 
            text="→ Skip (S)", 
            command=self.skip,
            bg='#FFC107',
            fg='black',
            font=('Arial', 16, 'bold'),
            height=2
        )
        self.skip_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        # Control frame
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.save_button = ttk.Button(self.control_frame, text="Save Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.stats_label = ttk.Label(self.control_frame, text="", font=('Arial', 10))
        self.stats_label.pack(side=tk.LEFT, padx=20)
        
        # Keyboard shortcuts
        self.root.bind('y', lambda e: self.label_yes())
        self.root.bind('n', lambda e: self.label_no())
        self.root.bind('s', lambda e: self.skip())
        self.root.bind('b', lambda e: self.go_back())
        self.root.bind('<space>', lambda e: self.skip())
        self.root.bind('<Escape>', lambda e: self.save_and_quit())
        self.root.bind('<Left>', lambda e: self.go_back())  # Also allow left arrow
        
        # Load first image
        self.show_current_image()
        
    def show_current_image(self):
        if self.current_idx >= len(self.images):
            self.show_completion()
            return
        
        img_data = self.images[self.current_idx]
        img_path = img_data['path']
        
        # Load and display image
        img = Image.open(img_path)
        
        # Resize to fit window while maintaining aspect ratio
        max_size = (900, 600)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Update info
        already_labeled = "✓ Already labeled" if img_path in self.labels else ""
        current_label = ""
        if img_path in self.labels:
            label_text = "VISIBLE" if self.labels[img_path] == 1 else "NOT VISIBLE"
            current_label = f" [Current: {label_text}]"
        
        self.progress_label.configure(
            text=f"Image {self.current_idx + 1:,} / {len(self.images):,}  {already_labeled}{current_label}"
        )
        
        self.folder_label.configure(
            text=f"Folder: {img_data['folder']} | Expected number: {img_data['folder_label']}"
        )
        
        # Update stats
        total_labeled = len(self.labels)
        yes_count = sum(1 for v in self.labels.values() if v == 1)
        no_count = sum(1 for v in self.labels.values() if v == 0)
        self.stats_label.configure(
            text=f"Total labeled: {total_labeled:,} | Yes: {yes_count:,} | No: {no_count:,} | This session: {self.session_count}"
        )
        
        # Enable/disable back button
        if self.current_idx > 0:
            self.back_button.config(state='normal')
        else:
            self.back_button.config(state='disabled')
    
    def label_yes(self):
        img_path = self.images[self.current_idx]['path']
        
        # Store history for undo
        old_label = self.labels.get(img_path, None)
        self.label_history.append({
            'idx': self.current_idx,
            'path': img_path,
            'old_label': old_label,
            'new_label': 1
        })
        
        self.labels[img_path] = 1  # 1 = visible
        self.session_count += 1
        self.next_image()
    
    def label_no(self):
        img_path = self.images[self.current_idx]['path']
        
        # Store history for undo
        old_label = self.labels.get(img_path, None)
        self.label_history.append({
            'idx': self.current_idx,
            'path': img_path,
            'old_label': old_label,
            'new_label': 0
        })
        
        self.labels[img_path] = 0  # 0 = not visible
        self.session_count += 1
        self.next_image()
    
    def skip(self):
        # Store history for undo (no label change)
        self.label_history.append({
            'idx': self.current_idx,
            'path': self.images[self.current_idx]['path'],
            'action': 'skip'
        })
        self.next_image()
    
    def go_back(self):
        """Go back to previous image and undo last action"""
        if self.current_idx <= 0:
            return
        
        # Undo last action if there is one
        if self.label_history:
            last_action = self.label_history.pop()
            
            # Restore old label state
            if 'action' not in last_action:  # Was a labeling action
                if last_action['old_label'] is None:
                    # Remove the label we just added
                    if last_action['path'] in self.labels:
                        del self.labels[last_action['path']]
                        self.session_count -= 1
                else:
                    # Restore previous label
                    self.labels[last_action['path']] = last_action['old_label']
                    self.session_count -= 1
        
        # Go back one image
        self.current_idx -= 1
        self.show_current_image()
    
    def next_image(self):
        self.current_idx += 1
        
        # Auto-save every 50 labels
        if self.session_count > 0 and self.session_count % 50 == 0:
            self.save_labels()
        
        self.show_current_image()
    
    def save_labels(self):
        with open(self.output_labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"Saved {len(self.labels)} labels to {self.output_labels_file}")
        self.stats_label.configure(
            text=f"✓ SAVED! Total labeled: {len(self.labels):,} | This session: {self.session_count}"
        )
    
    def save_and_quit(self):
        self.save_labels()
        self.root.quit()
    
    def show_completion(self):
        self.image_label.configure(text="🎉 All images labeled!", font=('Arial', 24))
        self.save_labels()
    
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    labeller = ImageLabeller(
        image_dir='C:/Temp/jersey-2023/train/images/',
        folder_labels_file='train_gt.json',
        output_labels_file='visibility_labels.json'
    )
    labeller.run()