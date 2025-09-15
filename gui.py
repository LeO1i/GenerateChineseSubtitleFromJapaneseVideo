import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import whisper
import torch
import subprocess
from speech_extract import JapaneseVideoSubtitleGenerator
from write_sutitle import WriteSubtitle


class SubtitleGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Japanese Subtitle Generator")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set theme and styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.video_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.selected_model = tk.StringVar(value="medium")
        self.processing = False
        
        # Check system requirements on startup
        self.check_system_requirements()
        
        # Available Whisper models
        self.whisper_models = [
            "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
        ]
        
        self.setup_ui()
        
    def on_closing(self):
        """Handle window close event"""
        if self.processing:
            # If processing is ongoing, ask user if they want to cancel
            result = messagebox.askyesno("Confirm", "Processing is in progress. Are you sure you want to exit?")
            if not result:
                return
        
        # Close the application
        self.root.quit()
        self.root.destroy()
        sys.exit(0)
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Japanese Subtitle Generator", 
                               font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # System info frame
        system_frame = ttk.LabelFrame(main_frame, text="System Information", padding="5")
        system_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU/CPU status
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        device_info = f"Processing device: {device}"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_info += f" ({gpu_name})"
            # Add CUDA version info
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            device_info += f" | CUDA: {cuda_version}"
        
        device_label = ttk.Label(system_frame, text=device_info, 
                                font=("Microsoft YaHei", 10, "bold"),
                                foreground="green" if torch.cuda.is_available() else "orange")
        device_label.pack(anchor=tk.W)
        
        # Performance note
        perf_note = "Note: GPU processing is much faster than CPU"
        if not torch.cuda.is_available():
            perf_note = "Note: CPU processing will be slower. Install CUDA for better performance."
        
        perf_label = ttk.Label(system_frame, text=perf_note, 
                              font=("Microsoft YaHei", 9), foreground="gray")
        perf_label.pack(anchor=tk.W)
        
        # Video file selection
        ttk.Label(main_frame, text="Video file:", font=("Microsoft YaHei", 9)).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50, font=("Microsoft YaHei", 9)).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video, style="TButton").grid(row=2, column=2, pady=5)
        
        # SRT file selection (for burning existing subtitles)
        ttk.Label(main_frame, text="SRT file (optional):", font=("Microsoft YaHei", 9)).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.srt_path, width=50, font=("Microsoft YaHei", 9)).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_srt, style="TButton").grid(row=3, column=2, pady=5)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output directory:", font=("Microsoft YaHei", 9)).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50, font=("Microsoft YaHei", 9)).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_dir, style="TButton").grid(row=4, column=2, pady=5)
        
        # Whisper model selection
        ttk.Label(main_frame, text="Whisper model:", font=("Microsoft YaHei", 9)).grid(row=5, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(main_frame, textvariable=self.selected_model, 
                                  values=self.whisper_models, state="readonly", width=20, font=("Microsoft YaHei", 9))
        model_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=5)
        
        # Model info
        model_info = ttk.Label(main_frame, text="Larger model = better accuracy but slower", 
                              font=("Microsoft YaHei", 8), foreground="gray")
        model_info.grid(row=5, column=2, sticky=tk.W, pady=5)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Generate subtitles button
        self.generate_btn = ttk.Button(button_frame, text="Generate Subtitles", 
                                      command=self.generate_subtitles, style="Accent.TButton")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Burn subtitles button
        self.burn_btn = ttk.Button(button_frame, text="Burn Subtitles to Video", 
                                  command=self.burn_subtitles)
        self.burn_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = ttk.Button(button_frame, text="Clear All", 
                                   command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Microsoft YaHei", 10))
        self.status_label.grid(row=9, column=0, columnspan=3, pady=5)
        
        # Log area
        ttk.Label(main_frame, text="Processing log:", font=("Microsoft YaHei", 9)).grid(row=10, column=0, sticky=tk.W, pady=(20, 5))
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, font=("Microsoft YaHei", 9))
        self.log_text.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for log area
        main_frame.rowconfigure(11, weight=1)
        
        # Initialize output directory to current directory
        self.output_dir.set(os.getcwd())
        
        # Add initial log message
        self.log_message("GUI initialized successfully")
        self.log_message(f"Processing device: {device}")
        if torch.cuda.is_available():
            self.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            self.log_message(f"CUDA version: {cuda_version}")
        else:
            self.log_message("Using CPU for processing")
            self.log_message("Tip: For GPU acceleration, install the CUDA build of PyTorch")
        
        # Show help button
        help_btn = ttk.Button(main_frame, text="Help", command=self.show_help)
        help_btn.grid(row=12, column=0, columnspan=3, pady=10)
        
    def browse_video(self):
        """Browse for video file"""
        filetypes = [
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"),
            ("MP4", "*.mp4"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=filetypes
        )
        if filename:
            self.video_path.set(filename)
            # Auto-set output directory to video directory if not set
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(filename))
    
    def browse_srt(self):
        """Browse for SRT file"""
        filetypes = [
            ("SRT files", "*.srt"),
            ("All files", "*.*")
        ]
        # Prefer opening at current SRT directory, else output directory, else CWD
        initial_dir = None
        try:
            if self.srt_path.get():
                initial_dir = os.path.dirname(self.srt_path.get())
            elif self.output_dir.get():
                initial_dir = self.output_dir.get()
            else:
                initial_dir = os.getcwd()
        except Exception:
            initial_dir = os.getcwd()
        filename = filedialog.askopenfilename(
            title="Select SRT file",
            filetypes=filetypes,
            initialdir=initial_dir
        )
        if filename:
            self.srt_path.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select output directory"
        )
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message):
        """Add message to log area"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def set_processing_state(self, processing):
        """Enable/disable buttons during processing"""
        self.processing = processing
        if processing:
            self.generate_btn.config(state='disabled')
            self.burn_btn.config(state='disabled')
            self.clear_btn.config(state='disabled')
            self.progress.start()
        else:
            self.generate_btn.config(state='normal')
            self.burn_btn.config(state='normal')
            self.clear_btn.config(state='normal')
            self.progress.stop()
    
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return False
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "The selected video file does not exist.")
            return False
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
        
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {e}")
                return False
        
        return True
    
    def generate_subtitles(self):
        """Generate subtitles from video"""
        if not self.validate_inputs():
            return
        
        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._generate_subtitles_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_subtitles_thread(self):
        """Thread function for subtitle generation"""
        try:
            self.set_processing_state(True)
            self.update_status("Initializing...")
            self.log_message("Start generating subtitles...")
            
            # Create generator with selected model
            self.log_message(f"Loading Whisper model: {self.selected_model.get()}")
            generator = JapaneseVideoSubtitleGenerator()
            
            # Override the model loading to use selected model
            generator.whisper_model = whisper.load_model(
                self.selected_model.get(), 
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            
            self.log_message("Model loaded successfully")
            
            # Process video
            self.update_status("Processing video...")
            srt_path = generator.process_video_to_srt(
                self.video_path.get(), 
                self.output_dir.get()
            )
            
            if srt_path:
                self.srt_path.set(srt_path)
                self.log_message("Subtitle generation completed!")
                self.log_message(f"Bilingual SRT file: {srt_path}")
                self.log_message("Note: Only Chinese lines will be burned when creating hardsubs")
                
                self.update_status("Subtitle generation completed")
                messagebox.showinfo("Success", 
                    f"Subtitles generated successfully!\n\n"
                    f"Bilingual SRT (JA + ZH):\n{srt_path}\n\n"
                    f"Note: Burning uses Chinese-only lines")
            else:
                self.log_message("Subtitle generation failed")
                self.update_status("Subtitle generation failed")
                messagebox.showerror("Error", "Subtitle generation failed. Please check logs.")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.update_status("Error occurred")
            messagebox.showerror("Error", f"Error occurred: {str(e)}")
        finally:
            self.set_processing_state(False)
    
    def burn_subtitles(self):
        """Burn subtitles to video"""
        if not self.validate_inputs():
            return
        
        if not self.srt_path.get():
            messagebox.showerror("Error", "Please select an SRT file or generate subtitles first.")
            return
        
        if not os.path.exists(self.srt_path.get()):
            messagebox.showerror("Error", "The selected SRT file does not exist.")
            return
        
        # Run in separate thread
        thread = threading.Thread(target=self._burn_subtitles_thread)
        thread.daemon = True
        thread.start()
    
    def _burn_subtitles_thread(self):
        """Thread function for burning subtitles"""
        try:
            self.set_processing_state(True)
            self.update_status("Burning subtitles...")
            self.log_message("Start burning subtitles...")
            
            # Create writer
            writer = WriteSubtitle()
            
            # Generate output path
            video_name = os.path.splitext(os.path.basename(self.video_path.get()))[0]
            output_video_path = os.path.join(
                self.output_dir.get(), 
                f"{video_name}_cn_hardsub.mp4"
            )
            
            self.log_message(f"Output video will be saved to: {output_video_path}")
            
            # Burn subtitles
            success = writer.burn_subtitles(
                self.video_path.get(),
                self.srt_path.get(),
                output_video_path
            )
            
            if success:
                self.log_message("Subtitle burning completed")
                self.update_status("Burning completed")
                messagebox.showinfo("Success", f"Video with subtitles saved to:\n{output_video_path}")
            else:
                self.log_message("Subtitle burning failed")
                self.update_status("Burning failed")
                messagebox.showerror("Error", "Subtitle burning failed. Please check logs.")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.update_status("Error occurred")
            messagebox.showerror("Error", f"Error occurred: {str(e)}")
        finally:
            self.set_processing_state(False)
    
    def clear_all(self):
        """Clear all inputs"""
        self.video_path.set("")
        self.srt_path.set("")
        self.output_dir.set(os.getcwd())
        self.selected_model.set("medium")
        self.log_text.delete(1.0, tk.END)
        self.update_status("Ready")
        
        # Add initial log message back
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        self.log_message("GUI reset successfully")
        self.log_message(f"Processing device: {device}")
        if torch.cuda.is_available():
            self.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.log_message("Using CPU for processing")
    
    def check_system_requirements(self):
        """Check system requirements for Windows"""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 9):
            issues.append("Python version is too low, Python 3.9 or later is required")
        
        # Check FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                issues.append("FFmpeg not found or not runnable")
        except FileNotFoundError:
            issues.append("FFmpeg is not installed or not in PATH")
        
        # Check required packages
        try:
            import whisper
            import googletrans
        except ImportError as e:
            issues.append(f"Missing required Python package: {e}")
        
        # Show warnings if any issues found
        if issues:
            warning_text = "Detected the following issues:\n\n" + "\n".join(f"• {issue}" for issue in issues)
            warning_text += "\n\nPlease run install.bat to resolve these issues."
            messagebox.showwarning("System Check", warning_text)
    
    def show_help(self):
        """Show help dialog with usage instructions"""
        help_text = """
Usage:

1. Select a video file
   - Supported formats: MP4, MKV, AVI, MOV, WMV, FLV
   - Clear audio yields better recognition quality

2. Choose a Whisper model
   - tiny/base: Fast but lower accuracy
   - small/medium: Balanced speed and accuracy (recommended)
   - large/large-v2/large-v3: Higher accuracy but slower

3. Generate subtitles
   - Click "Generate Subtitles" to start
   - Processing time depends on video length and model
   - The generated SRT will be saved in the output directory

4. Burn subtitles
   - Select the generated SRT file
   - Click "Burn Subtitles to Video"
   - The new video will include hardcoded Chinese subtitles

Notes:
• First run downloads the Whisper model; internet required
• GPU is much faster than CPU; consider installing CUDA
• Ensure enough disk space for temporary files
• Be patient with large files

Troubleshooting:
• Check system requirements if errors occur
• Ensure FFmpeg is properly installed
• Check network connectivity (for translation service)
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.resizable(True, True)
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=("Microsoft YaHei", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SubtitleGeneratorGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
