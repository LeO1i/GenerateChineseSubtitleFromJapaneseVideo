import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import torch
import subprocess
from ffmpeg_utils import subprocess_kwargs as _subprocess_kwargs
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
        self.selected_asr_model = tk.StringVar(value="Qwen/Qwen3-ASR-1.7B")
        self.selected_mt_model = tk.StringVar(value="tencent/HY-MT1.5-1.8B")
        self.chunk_size_seconds = tk.StringVar(value="120")
        self.overlap_seconds = tk.StringVar(value="1.5")
        self.quality_mode = tk.StringVar(value="fast")
        self.glossary_path = tk.StringVar()
        self.use_advanced_mt = tk.BooleanVar(value=False)
        # Default preference is GPU; runtime falls back to CPU if CUDA is unavailable.
        self.device_preference = tk.StringVar(value="cuda:0")
        self.processing = False
        
        # Check system requirements on startup
        self.check_system_requirements()
        
        # Available model presets
        self.asr_models = [
            "Qwen/Qwen3-ASR-1.7B",
        ]
        self.mt_models = [
            "tencent/HY-MT1.5-1.8B",
        ]
        
        self.setup_ui()

    def _run_on_ui_thread(self, func, *args, wait=False):
        """Run UI operations safely from worker threads."""
        if threading.current_thread() is threading.main_thread():
            return func(*args)

        if not wait:
            self.root.after(0, lambda: func(*args))
            return None

        done = threading.Event()
        result = {"value": None}

        def _wrapped():
            try:
                result["value"] = func(*args)
            finally:
                done.set()

        self.root.after(0, _wrapped)
        done.wait()
        return result["value"]
        
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
        active_device = "CUDA GPU" if torch.cuda.is_available() else "CPU (fallback)"
        device_info = f"Device preference: CUDA GPU (default) | Active: {active_device}"
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
        
        # ASR model selection
        ttk.Label(main_frame, text="ASR model:", font=("Microsoft YaHei", 9)).grid(row=5, column=0, sticky=tk.W, pady=5)
        asr_combo = ttk.Combobox(
            main_frame,
            textvariable=self.selected_asr_model,
            values=self.asr_models,
            state="readonly",
            width=36,
            font=("Microsoft YaHei", 9),
        )
        asr_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=5)

        # MT model selection
        ttk.Label(main_frame, text="MT model:", font=("Microsoft YaHei", 9)).grid(row=6, column=0, sticky=tk.W, pady=5)
        mt_combo = ttk.Combobox(
            main_frame,
            textvariable=self.selected_mt_model,
            values=self.mt_models,
            state="readonly",
            width=36,
            font=("Microsoft YaHei", 9),
        )
        mt_combo.grid(row=6, column=1, sticky=tk.W, padx=(5, 5), pady=5)

        ttk.Checkbutton(
            main_frame,
            text="Enable advanced MT fallback chain (try 7B first)",
            variable=self.use_advanced_mt,
        ).grid(row=6, column=2, sticky=tk.W, pady=5)

        # Chunk and quality settings
        ttk.Label(main_frame, text="Chunk size (30-600s):", font=("Microsoft YaHei", 9)).grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.chunk_size_seconds, width=10, font=("Microsoft YaHei", 9)).grid(
            row=7, column=1, sticky=tk.W, padx=(5, 5), pady=5
        )
        ttk.Label(main_frame, text="Default: 120", font=("Microsoft YaHei", 8), foreground="gray").grid(
            row=7, column=2, sticky=tk.W, pady=5
        )

        ttk.Label(main_frame, text="Overlap seconds:", font=("Microsoft YaHei", 9)).grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.overlap_seconds, width=10, font=("Microsoft YaHei", 9)).grid(
            row=8, column=1, sticky=tk.W, padx=(5, 5), pady=5
        )
        ttk.Combobox(
            main_frame,
            textvariable=self.quality_mode,
            values=["fast", "accurate"],
            state="readonly",
            width=12,
            font=("Microsoft YaHei", 9),
        ).grid(row=8, column=2, sticky=tk.W, pady=5)

        # Optional glossary
        ttk.Label(main_frame, text="Glossary file (optional):", font=("Microsoft YaHei", 9)).grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.glossary_path, width=50, font=("Microsoft YaHei", 9)).grid(
            row=9, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5
        )
        ttk.Button(main_frame, text="Browse", command=self.browse_glossary, style="TButton").grid(row=9, column=2, pady=5)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=11, column=0, columnspan=3, pady=10)
        
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
        self.progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", font=("Microsoft YaHei", 10))
        self.status_label.grid(row=13, column=0, columnspan=3, pady=5)
        
        # Log area
        ttk.Label(main_frame, text="Processing log:", font=("Microsoft YaHei", 9)).grid(row=14, column=0, sticky=tk.W, pady=(20, 5))
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, font=("Microsoft YaHei", 9))
        self.log_text.grid(row=15, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for log area
        main_frame.rowconfigure(15, weight=1)
        
        # Initialize output directory to current directory
        self.output_dir.set(os.getcwd())
        
        # Add initial log message
        self.log_message("GUI initialized successfully")
        self._log_device_info()
        
        # Show help button
        help_btn = ttk.Button(main_frame, text="Help", command=self.show_help)
        help_btn.grid(row=16, column=0, columnspan=3, pady=10)
        
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

    def browse_glossary(self):
        """Browse for optional glossary file"""
        filetypes = [
            ("Text files", "*.txt *.tsv *.csv"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Select glossary file",
            filetypes=filetypes,
        )
        if filename:
            self.glossary_path.set(filename)
    
    def log_message(self, message):
        """Add message to log area"""
        def _update():
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()

        self._run_on_ui_thread(_update, wait=True)
    
    def update_status(self, message):
        """Update status label"""
        def _update():
            self.status_label.config(text=message)
            self.root.update_idletasks()

        self._run_on_ui_thread(_update, wait=True)
    
    def set_processing_state(self, processing):
        """Enable/disable buttons during processing"""
        def _update():
            self.processing = processing
            if processing:
                self.generate_btn.config(state='disabled')
                self.burn_btn.config(state='disabled')
                self.clear_btn.config(state='disabled')
                self.progress.configure(value=0)
            else:
                self.generate_btn.config(state='normal')
                self.burn_btn.config(state='normal')
                self.clear_btn.config(state='normal')
                self.progress.configure(value=100)

        self._run_on_ui_thread(_update, wait=True)

    def reset_progress(self):
        def _update():
            self.progress.configure(value=0)

        self._run_on_ui_thread(_update, wait=True)

    def update_progress(self, value, message=None):
        def _update():
            self.progress.configure(value=max(0.0, min(100.0, float(value))))
            if message:
                self.status_label.config(text=message)
            self.root.update_idletasks()

        self._run_on_ui_thread(_update, wait=True)

    def show_info(self, title, message):
        """Thread-safe info dialog."""
        return self._run_on_ui_thread(messagebox.showinfo, title, message, wait=True)

    def show_error(self, title, message):
        """Thread-safe error dialog."""
        return self._run_on_ui_thread(messagebox.showerror, title, message, wait=True)
    
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

        try:
            chunk_size = int(self.chunk_size_seconds.get().strip())
            if chunk_size < 30 or chunk_size > 600:
                raise ValueError("Chunk size must be 30-600")
        except Exception:
            messagebox.showerror("Error", "Chunk size must be an integer between 30 and 600.")
            return False

        try:
            overlap = float(self.overlap_seconds.get().strip())
            if overlap < 0 or overlap > 10:
                raise ValueError("Overlap must be 0-10")
        except Exception:
            messagebox.showerror("Error", "Overlap seconds must be a number between 0 and 10.")
            return False

        glossary_file = self.glossary_path.get().strip()
        if glossary_file and not os.path.exists(glossary_file):
            messagebox.showerror("Error", "Glossary file does not exist.")
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
            self.reset_progress()
            self.update_status("Initializing...")
            self.log_message("Start generating subtitles...")
            
            # Create generator with selected models
            self.log_message(f"Loading ASR model: {self.selected_asr_model.get()}")
            self.log_message(f"Loading MT model: {self.selected_mt_model.get()}")
            generator = JapaneseVideoSubtitleGenerator(
                asr_model_id=self.selected_asr_model.get(),
                mt_model_id=self.selected_mt_model.get(),
                use_advanced_mt=self.use_advanced_mt.get(),
                quality_mode=self.quality_mode.get(),
                glossary_path=self.glossary_path.get().strip() or None,
                device=self.device_preference.get(),
                progress_callback=self.update_progress,
            )
            
            self.log_message("Model loaded successfully")
            
            # Process video
            self.update_status("Processing video...")
            self.update_progress(5, "Processing video...")
            srt_path = generator.process_video(
                self.video_path.get(),
                self.output_dir.get(),
                chunk_size_seconds=int(self.chunk_size_seconds.get().strip()),
                overlap_seconds=float(self.overlap_seconds.get().strip()),
                quality_mode=self.quality_mode.get(),
                glossary_path=self.glossary_path.get().strip() or None,
            )
            
            if srt_path:
                self.update_progress(100, "Subtitle generation completed")
                self.srt_path.set(srt_path)
                self.log_message("Subtitle generation completed!")
                self.log_message(f"Bilingual SRT file: {srt_path}")
                self.log_message("Note: Only Chinese lines will be burned when creating hardsubs")
                
                self.update_status("Subtitle generation completed")
                self.show_info("Success",
                    f"Subtitles generated successfully!\n\n"
                    f"Bilingual SRT (JA + ZH):\n{srt_path}\n\n"
                    f"Note: Burning uses Chinese-only lines")
            else:
                self.log_message("Subtitle generation failed")
                self.update_status("Subtitle generation failed")
                self.show_error("Error", "Subtitle generation failed. Please check logs.")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.update_status("Error occurred")
            self.show_error("Error", f"Error occurred: {str(e)}")
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
            self.reset_progress()
            self.update_status("Burning subtitles...")
            self.update_progress(10, "Preparing subtitle burn...")
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
                self.update_progress(100, "Burning completed")
                self.log_message("Subtitle burning completed")
                self.update_status("Burning completed")
                self.show_info("Success", f"Video with subtitles saved to:\n{output_video_path}")
            else:
                self.update_progress(100, "Burning failed")
                self.log_message("Subtitle burning failed")
                self.update_status("Burning failed")
                self.show_error("Error", "Subtitle burning failed. Please check logs.")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.update_status("Error occurred")
            self.show_error("Error", f"Error occurred: {str(e)}")
        finally:
            self.set_processing_state(False)
    
    def clear_all(self):
        """Clear all inputs"""
        self.video_path.set("")
        self.srt_path.set("")
        self.output_dir.set(os.getcwd())
        self.selected_asr_model.set("Qwen/Qwen3-ASR-1.7B")
        self.selected_mt_model.set("tencent/HY-MT1.5-1.8B")
        self.chunk_size_seconds.set("120")
        self.overlap_seconds.set("1.5")
        self.quality_mode.set("fast")
        self.glossary_path.set("")
        self.use_advanced_mt.set(False)
        self.device_preference.set("cuda:0")
        self.log_text.delete(1.0, tk.END)
        self.update_status("Ready")
        
        # Add initial log message back
        self.log_message("GUI reset successfully")
        self._log_device_info()
    
    def check_system_requirements(self):
        """Check system requirements for Windows"""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 9):
            issues.append("Python version is too low, Python 3.9 or later is required")
        
        # Check FFmpeg
        try:
            run_kwargs = {"capture_output": True, "text": True, **_subprocess_kwargs()}
            result = subprocess.run(['ffmpeg', '-version'], **run_kwargs)
            if result.returncode != 0:
                issues.append("FFmpeg not found or not runnable")
        except FileNotFoundError:
            issues.append("FFmpeg is not installed or not in PATH")
        
        # Check required packages
        try:
            import transformers
        except ImportError as e:
            issues.append(f"Missing required Python package: {e}")
        
        # Show warnings if any issues found
        if issues:
            warning_text = "Detected the following issues:\n\n" + "\n".join(f"• {issue}" for issue in issues)
            warning_text += "\n\nPlease run install.bat to resolve these issues."
            messagebox.showwarning("System Check", warning_text)
    
    def _log_device_info(self):
        """Log device/GPU information to the log area."""
        self.log_message("Device preference: CUDA GPU (cuda:0)")
        if torch.cuda.is_available():
            self.log_message("Active device: CUDA GPU")
            self.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            self.log_message(f"CUDA version: {cuda_version}")
        else:
            self.log_message("Active device: CPU fallback")
            self.log_message("Tip: For GPU acceleration, install the CUDA build of PyTorch")

    def show_help(self):
        """Show help dialog with usage instructions"""
        help_text = """
Usage:

1. Select a video file
   - Supported formats: MP4, MKV, AVI, MOV, WMV, FLV
   - Clear audio yields better recognition quality

2. Select model and quality options
   - ASR defaults to Qwen3-ASR-1.7B
   - MT defaults to HY-MT compatible path (ja->zh)
   - Quality: fast or accurate
   - Chunk size must be 30-600 seconds (default 120)
   - Overlap defaults to 1.5 seconds

3. Generate subtitles
   - Click "Generate Subtitles" to start
   - Processing time depends on video length, chunk size, and model
   - The generated SRT will be saved in the output directory

4. Burn subtitles
   - Select the generated SRT file
   - Click "Burn Subtitles to Video"
   - The new video will include hardcoded Chinese subtitles

Notes:
• First run downloads models from Hugging Face; internet required
• GPU is much faster than CPU; consider installing CUDA
• Ensure enough disk space for temporary files
• Be patient with large files

Troubleshooting:
• Check system requirements if errors occur
• Ensure FFmpeg is properly installed
• Check network connectivity (for model download)
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
