import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import whisper
import torch
from speech_extract import JapaneseVideoSubtitleGenerator
from write_sutitle import WriteSubtitle


class SubtitleGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("日语视频字幕生成器")
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
        
        # Available Whisper models
        self.whisper_models = [
            "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
        ]
        
        self.setup_ui()
        
    def on_closing(self):
        """Handle window close event"""
        if self.processing:
            # If processing is ongoing, ask user if they want to cancel
            result = messagebox.askyesno("确认", "正在处理中，确定要退出吗？")
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
        title_label = ttk.Label(main_frame, text="日语视频字幕生成器", 
                               font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # System info frame
        system_frame = ttk.LabelFrame(main_frame, text="系统信息", padding="5")
        system_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU/CPU status
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        device_info = f"处理设备: {device}"
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
        perf_note = "注意: GPU处理比CPU快很多"
        if not torch.cuda.is_available():
            perf_note = "注意: CPU处理会比较慢。建议安装CUDA以获得更好的性能。"
        
        perf_label = ttk.Label(system_frame, text=perf_note, 
                              font=("Microsoft YaHei", 9), foreground="gray")
        perf_label.pack(anchor=tk.W)
        
        # Video file selection
        ttk.Label(main_frame, text="视频文件:", font=("Microsoft YaHei", 9)).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50, font=("Microsoft YaHei", 9)).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_video, style="TButton").grid(row=2, column=2, pady=5)
        
        # SRT file selection (for burning existing subtitles)
        ttk.Label(main_frame, text="SRT文件 (可选):", font=("Microsoft YaHei", 9)).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.srt_path, width=50, font=("Microsoft YaHei", 9)).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_srt, style="TButton").grid(row=3, column=2, pady=5)
        
        # Output directory selection
        ttk.Label(main_frame, text="输出目录:", font=("Microsoft YaHei", 9)).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50, font=("Microsoft YaHei", 9)).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_output_dir, style="TButton").grid(row=4, column=2, pady=5)
        
        # Whisper model selection
        ttk.Label(main_frame, text="Whisper模型:", font=("Microsoft YaHei", 9)).grid(row=5, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(main_frame, textvariable=self.selected_model, 
                                  values=self.whisper_models, state="readonly", width=20, font=("Microsoft YaHei", 9))
        model_combo.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=5)
        
        # Model info
        model_info = ttk.Label(main_frame, text="更大的模型 = 更好的准确性但处理更慢", 
                              font=("Microsoft YaHei", 8), foreground="gray")
        model_info.grid(row=5, column=2, sticky=tk.W, pady=5)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Generate subtitles button
        self.generate_btn = ttk.Button(button_frame, text="生成字幕", 
                                      command=self.generate_subtitles, style="Accent.TButton")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Burn subtitles button
        self.burn_btn = ttk.Button(button_frame, text="烧录字幕到视频", 
                                  command=self.burn_subtitles)
        self.burn_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = ttk.Button(button_frame, text="清除所有", 
                                   command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="就绪", font=("Microsoft YaHei", 10))
        self.status_label.grid(row=9, column=0, columnspan=3, pady=5)
        
        # Log area
        ttk.Label(main_frame, text="处理日志:", font=("Microsoft YaHei", 9)).grid(row=10, column=0, sticky=tk.W, pady=(20, 5))
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, font=("Microsoft YaHei", 9))
        self.log_text.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for log area
        main_frame.rowconfigure(11, weight=1)
        
        # Initialize output directory to current directory
        self.output_dir.set(os.getcwd())
        
        # Add initial log message
        self.log_message("GUI初始化成功")
        self.log_message(f"处理设备: {device}")
        if torch.cuda.is_available():
            self.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            self.log_message(f"CUDA版本: {cuda_version}")
        else:
            self.log_message("使用CPU进行处理")
            self.log_message("提示: 如需GPU加速，请安装CUDA版本的PyTorch")
        
    def browse_video(self):
        """Browse for video file"""
        filetypes = [
            ("视频文件", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"),
            ("MP4文件", "*.mp4"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择视频文件",
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
            ("SRT文件", "*.srt"),
            ("所有文件", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="选择SRT文件",
            filetypes=filetypes
        )
        if filename:
            self.srt_path.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="选择输出目录"
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
            messagebox.showerror("错误", "请选择视频文件。")
            return False
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("错误", "选择的视频文件不存在。")
            return False
        
        if not self.output_dir.get():
            messagebox.showerror("错误", "请选择输出目录。")
            return False
        
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {e}")
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
            self.update_status("初始化中...")
            self.log_message("开始生成字幕...")
            
            # Create generator with selected model
            self.log_message(f"加载Whisper模型: {self.selected_model.get()}")
            generator = JapaneseVideoSubtitleGenerator()
            
            # Override the model loading to use selected model
            generator.whisper_model = whisper.load_model(
                self.selected_model.get(), 
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            
            self.log_message("模型加载成功")
            
            # Process video
            self.update_status("处理视频中...")
            srt_path = generator.process_video_to_srt(
                self.video_path.get(), 
                self.output_dir.get()
            )
            
            if srt_path:
                self.srt_path.set(srt_path)
                self.log_message("字幕生成完成!")
                self.log_message(f"双语SRT文件: {srt_path}")
                self.log_message("注意: 烧录时会自动提取中文部分")
                
                self.update_status("字幕生成完成")
                messagebox.showinfo("成功", 
                    f"字幕生成成功!\n\n"
                    f"双语SRT文件 (中日文):\n{srt_path}\n\n"
                    f"烧录时会自动提取中文部分")
            else:
                self.log_message("字幕生成失败")
                self.update_status("字幕生成失败")
                messagebox.showerror("错误", "字幕生成失败。请查看日志了解详情。")
                
        except Exception as e:
            self.log_message(f"错误: {str(e)}")
            self.update_status("发生错误")
            messagebox.showerror("错误", f"发生错误: {str(e)}")
        finally:
            self.set_processing_state(False)
    
    def burn_subtitles(self):
        """Burn subtitles to video"""
        if not self.validate_inputs():
            return
        
        if not self.srt_path.get():
            messagebox.showerror("错误", "请选择SRT文件或先生成字幕。")
            return
        
        if not os.path.exists(self.srt_path.get()):
            messagebox.showerror("错误", "选择的SRT文件不存在。")
            return
        
        # Run in separate thread
        thread = threading.Thread(target=self._burn_subtitles_thread)
        thread.daemon = True
        thread.start()
    
    def _burn_subtitles_thread(self):
        """Thread function for burning subtitles"""
        try:
            self.set_processing_state(True)
            self.update_status("烧录字幕中...")
            self.log_message("开始烧录字幕...")
            
            # Create writer
            writer = WriteSubtitle()
            
            # Generate output path
            video_name = os.path.splitext(os.path.basename(self.video_path.get()))[0]
            output_video_path = os.path.join(
                self.output_dir.get(), 
                f"{video_name}_cn_hardsub.mp4"
            )
            
            self.log_message(f"输出视频将保存到: {output_video_path}")
            
            # Burn subtitles
            success = writer.burn_subtitles(
                self.video_path.get(),
                self.srt_path.get(),
                output_video_path
            )
            
            if success:
                self.log_message("字幕烧录完成")
                self.update_status("烧录完成")
                messagebox.showinfo("成功", f"带字幕的视频已保存到:\n{output_video_path}")
            else:
                self.log_message("字幕烧录失败")
                self.update_status("烧录失败")
                messagebox.showerror("错误", "字幕烧录失败。请查看日志了解详情。")
                
        except Exception as e:
            self.log_message(f"错误: {str(e)}")
            self.update_status("发生错误")
            messagebox.showerror("错误", f"发生错误: {str(e)}")
        finally:
            self.set_processing_state(False)
    
    def clear_all(self):
        """Clear all inputs"""
        self.video_path.set("")
        self.srt_path.set("")
        self.output_dir.set(os.getcwd())
        self.selected_model.set("medium")
        self.log_text.delete(1.0, tk.END)
        self.update_status("就绪")
        
        # Add initial log message back
        device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        self.log_message("GUI重置成功")
        self.log_message(f"处理设备: {device}")
        if torch.cuda.is_available():
            self.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.log_message("使用CPU进行处理")


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
