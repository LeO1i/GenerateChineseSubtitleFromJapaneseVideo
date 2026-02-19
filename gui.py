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
        self.root.title("日语字幕生成器")
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
        self.overlap_seconds = tk.StringVar(value="2")
        # UI 显示使用中文，运行时映射到 "fast"/"accurate"
        self.quality_mode = tk.StringVar(value="快速（fast）")
        self.audio_preset = tk.StringVar(value="标准（standard）")
        self.glossary_path = tk.StringVar()
        self.asr_terms_path = tk.StringVar()
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

    def _quality_mode_value(self):
        """Map UI selection to internal values expected by the pipeline."""
        raw = (self.quality_mode.get() or "").strip().lower()
        if "accurate" in raw or "精确" in raw or "准确" in raw:
            return "accurate"
        return "fast"

    def _audio_preset_value(self):
        raw = (self.audio_preset.get() or "").strip().lower()
        if "aggressive" in raw or "强力" in raw:
            return "aggressive"
        if "denoise" in raw or "降噪" in raw:
            return "denoise"
        return "standard"

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
        title_label = ttk.Label(main_frame, text="日语字幕生成器", 
                               font=("Microsoft YaHei", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # System info frame
        system_frame = ttk.LabelFrame(main_frame, text="系统信息", padding="5")
        system_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU/CPU status
        active_device = "CUDA 显卡" if torch.cuda.is_available() else "CPU（回退）"
        device_info = f"设备偏好：CUDA 显卡（默认） | 当前：{active_device}"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            device_info += f"（{gpu_name}）"
            # Add CUDA version info
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            device_info += f" | CUDA：{cuda_version}"
        
        device_label = ttk.Label(system_frame, text=device_info, 
                                font=("Microsoft YaHei", 10, "bold"),
                                foreground="green" if torch.cuda.is_available() else "orange")
        device_label.pack(anchor=tk.W)
        
        # Performance note
        perf_note = "提示：GPU 处理速度远快于 CPU"
        if not torch.cuda.is_available():
            perf_note = "提示：当前使用 CPU 处理会更慢。安装 CUDA 可获得更好的性能。"
        
        perf_label = ttk.Label(system_frame, text=perf_note, 
                              font=("Microsoft YaHei", 9), foreground="gray")
        perf_label.pack(anchor=tk.W)
        
        # Video file selection
        ttk.Label(main_frame, text="视频文件：", font=("Microsoft YaHei", 9)).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=50, font=("Microsoft YaHei", 9)).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_video, style="TButton").grid(row=2, column=2, pady=5)
        
        # SRT file selection (for burning existing subtitles)
        ttk.Label(main_frame, text="SRT 字幕文件（可选）：", font=("Microsoft YaHei", 9)).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.srt_path, width=50, font=("Microsoft YaHei", 9)).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_srt, style="TButton").grid(row=3, column=2, pady=5)
        
        # Output directory selection
        ttk.Label(main_frame, text="输出目录：", font=("Microsoft YaHei", 9)).grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50, font=("Microsoft YaHei", 9)).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output_dir, style="TButton").grid(row=4, column=2, pady=5)
        
        # ASR model selection
        ttk.Label(main_frame, text="ASR 模型：", font=("Microsoft YaHei", 9)).grid(row=5, column=0, sticky=tk.W, pady=5)
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
        ttk.Label(main_frame, text="MT 模型：", font=("Microsoft YaHei", 9)).grid(row=6, column=0, sticky=tk.W, pady=5)
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
            text="启用高级翻译回退链（优先尝试 7B）",
            variable=self.use_advanced_mt,
        ).grid(row=6, column=2, sticky=tk.W, pady=5)

        # Chunk and quality settings
        ttk.Label(main_frame, text="分块时长（30-600 秒）：", font=("Microsoft YaHei", 9)).grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.chunk_size_seconds, width=10, font=("Microsoft YaHei", 9)).grid(
            row=7, column=1, sticky=tk.W, padx=(5, 5), pady=5
        )
        ttk.Label(main_frame, text="默认：120", font=("Microsoft YaHei", 8), foreground="gray").grid(
            row=7, column=2, sticky=tk.W, pady=5
        )

        ttk.Label(main_frame, text="重叠时长（秒）：", font=("Microsoft YaHei", 9)).grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.overlap_seconds, width=10, font=("Microsoft YaHei", 9)).grid(
            row=8, column=1, sticky=tk.W, padx=(5, 5), pady=5
        )
        ttk.Combobox(
            main_frame,
            textvariable=self.quality_mode,
            values=["快速（fast）", "精确（accurate）"],
            state="readonly",
            width=12,
            font=("Microsoft YaHei", 9),
        ).grid(row=8, column=2, sticky=tk.W, pady=5)

        # Optional glossary
        ttk.Label(main_frame, text="术语表文件（可选）：", font=("Microsoft YaHei", 9)).grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.glossary_path, width=50, font=("Microsoft YaHei", 9)).grid(
            row=9, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5
        )
        ttk.Button(main_frame, text="浏览...", command=self.browse_glossary, style="TButton").grid(row=9, column=2, pady=5)

        ttk.Label(main_frame, text="ASR 术语/修正文件（可选）：", font=("Microsoft YaHei", 9)).grid(
            row=10, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.asr_terms_path, width=50, font=("Microsoft YaHei", 9)).grid(
            row=10, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5
        )
        ttk.Button(main_frame, text="浏览...", command=self.browse_asr_terms, style="TButton").grid(
            row=10, column=2, pady=5
        )

        ttk.Label(main_frame, text="音频增强预设：", font=("Microsoft YaHei", 9)).grid(row=11, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            main_frame,
            textvariable=self.audio_preset,
            values=["标准（standard）", "降噪（denoise）", "强力降噪（aggressive）"],
            state="readonly",
            width=18,
            font=("Microsoft YaHei", 9),
        ).grid(row=11, column=1, sticky=tk.W, padx=(5, 5), pady=5)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=13, column=0, columnspan=3, pady=10)
        
        # Generate subtitles button
        self.generate_btn = ttk.Button(button_frame, text="生成字幕", 
                                      command=self.generate_subtitles, style="Accent.TButton")
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Burn subtitles button
        self.burn_btn = ttk.Button(button_frame, text="烧录字幕", 
                                  command=self.burn_subtitles)
        self.burn_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = ttk.Button(button_frame, text="清空", 
                                   command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress.grid(row=14, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="就绪", font=("Microsoft YaHei", 10))
        self.status_label.grid(row=15, column=0, columnspan=3, pady=5)
        
        # Log area
        ttk.Label(main_frame, text="处理日志：", font=("Microsoft YaHei", 9)).grid(row=16, column=0, sticky=tk.W, pady=(20, 5))
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, font=("Microsoft YaHei", 9))
        self.log_text.grid(row=17, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for log area
        main_frame.rowconfigure(17, weight=1)
        
        # Initialize output directory to current directory
        self.output_dir.set(os.getcwd())
        
        # Add initial log message
        self.log_message("界面初始化完成")
        self._log_device_info()
        
        # Show help button
        help_btn = ttk.Button(main_frame, text="帮助", command=self.show_help)
        help_btn.grid(row=18, column=0, columnspan=3, pady=10)
        
    def browse_video(self):
        """Browse for video file"""
        filetypes = [
            ("视频文件", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"),
            ("MP4", "*.mp4"),
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
            ("SRT 字幕文件", "*.srt"),
            ("所有文件", "*.*")
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
            title="选择 SRT 字幕文件",
            filetypes=filetypes,
            initialdir=initial_dir
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

    def browse_glossary(self):
        """Browse for optional glossary file"""
        filetypes = [
            ("文本文件", "*.txt *.tsv *.csv"),
            ("所有文件", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="选择术语表文件",
            filetypes=filetypes,
        )
        if filename:
            self.glossary_path.set(filename)

    def browse_asr_terms(self):
        """Browse for optional ASR term/correction file."""
        filetypes = [
            ("文本文件", "*.txt *.tsv *.csv"),
            ("所有文件", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="选择 ASR 术语/修正文件",
            filetypes=filetypes,
        )
        if filename:
            self.asr_terms_path.set(filename)
    
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
            messagebox.showerror("错误", "请选择视频文件。")
            return False
        
        if not os.path.exists(self.video_path.get()):
            messagebox.showerror("错误", "所选视频文件不存在。")
            return False
        
        if not self.output_dir.get():
            messagebox.showerror("错误", "请选择输出目录。")
            return False
        
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录：{e}")
                return False

        try:
            chunk_size = int(self.chunk_size_seconds.get().strip())
            if chunk_size < 30 or chunk_size > 600:
                raise ValueError("Chunk size must be 30-600")
        except Exception:
            messagebox.showerror("错误", "分块时长必须是 30 到 600 之间的整数。")
            return False

        try:
            overlap = float(self.overlap_seconds.get().strip())
            if overlap < 0 or overlap > 10:
                raise ValueError("Overlap must be 0-10")
        except Exception:
            messagebox.showerror("错误", "重叠时长必须是 0 到 10 之间的数字。")
            return False

        glossary_file = self.glossary_path.get().strip()
        if glossary_file and not os.path.exists(glossary_file):
            messagebox.showerror("错误", "术语表文件不存在。")
            return False
        asr_terms_file = self.asr_terms_path.get().strip()
        if asr_terms_file and not os.path.exists(asr_terms_file):
            messagebox.showerror("错误", "ASR 术语/修正文件不存在。")
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
            self.update_status("初始化中...")
            self.log_message("开始生成字幕...")
            
            # Create generator with selected models
            self.log_message(f"加载 ASR 模型：{self.selected_asr_model.get()}")
            self.log_message(f"加载 MT 模型：{self.selected_mt_model.get()}")
            generator = JapaneseVideoSubtitleGenerator(
                asr_model_id=self.selected_asr_model.get(),
                mt_model_id=self.selected_mt_model.get(),
                use_advanced_mt=self.use_advanced_mt.get(),
                quality_mode=self._quality_mode_value(),
                glossary_path=self.glossary_path.get().strip() or None,
                asr_terms_path=self.asr_terms_path.get().strip() or None,
                audio_preset=self._audio_preset_value(),
                device=self.device_preference.get(),
                progress_callback=self.update_progress,
            )
            
            self.log_message("模型加载完成")
            
            # Process video
            self.update_status("正在处理视频...")
            self.update_progress(5, "正在处理视频...")
            srt_path = generator.process_video(
                self.video_path.get(),
                self.output_dir.get(),
                chunk_size_seconds=int(self.chunk_size_seconds.get().strip()),
                overlap_seconds=float(self.overlap_seconds.get().strip()),
                quality_mode=self._quality_mode_value(),
                glossary_path=self.glossary_path.get().strip() or None,
                asr_terms_path=self.asr_terms_path.get().strip() or None,
                audio_preset=self._audio_preset_value(),
            )
            
            if srt_path:
                self.update_progress(100, "字幕生成完成")
                self.srt_path.set(srt_path)
                self.log_message("字幕生成完成！")
                self.log_message(f"双语 SRT 文件：{srt_path}")
                self.log_message("提示：烧录硬字幕时只会使用中文行")
                
                self.update_status("字幕生成完成")
                self.show_info("成功",
                    f"字幕生成成功！\n\n"
                    f"双语 SRT（日语 + 中文）：\n{srt_path}\n\n"
                    f"提示：烧录时只使用中文行")
            else:
                self.log_message("字幕生成失败")
                self.update_status("字幕生成失败")
                self.show_error("错误", "字幕生成失败，请查看日志。")
                
        except Exception as e:
            self.log_message(f"错误：{str(e)}")
            self.update_status("发生错误")
            self.show_error("错误", f"发生错误：{str(e)}")
        finally:
            self.set_processing_state(False)
    
    def burn_subtitles(self):
        """Burn subtitles to video"""
        if not self.validate_inputs():
            return
        
        if not self.srt_path.get():
            messagebox.showerror("错误", "请先选择 SRT 字幕文件，或先生成字幕。")
            return
        
        if not os.path.exists(self.srt_path.get()):
            messagebox.showerror("错误", "所选 SRT 字幕文件不存在。")
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
            self.update_status("正在烧录字幕...")
            self.update_progress(10, "准备烧录...")
            self.log_message("开始烧录字幕...")
            
            # Create writer
            writer = WriteSubtitle()
            
            # Generate output path
            video_name = os.path.splitext(os.path.basename(self.video_path.get()))[0]
            output_video_path = os.path.join(
                self.output_dir.get(), 
                f"{video_name}_cn_hardsub.mp4"
            )
            
            self.log_message(f"输出视频将保存到：{output_video_path}")
            
            # Burn subtitles
            success = writer.burn_subtitles(
                self.video_path.get(),
                self.srt_path.get(),
                output_video_path
            )
            
            if success:
                self.update_progress(100, "烧录完成")
                self.log_message("字幕烧录完成")
                self.update_status("烧录完成")
                self.show_info("成功", f"带字幕的视频已保存到：\n{output_video_path}")
            else:
                self.update_progress(100, "烧录失败")
                self.log_message("字幕烧录失败")
                self.update_status("烧录失败")
                self.show_error("错误", "字幕烧录失败，请查看日志。")
                
        except Exception as e:
            self.log_message(f"错误：{str(e)}")
            self.update_status("发生错误")
            self.show_error("错误", f"发生错误：{str(e)}")
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
        self.overlap_seconds.set("2")
        self.quality_mode.set("快速（fast）")
        self.audio_preset.set("标准（standard）")
        self.glossary_path.set("")
        self.asr_terms_path.set("")
        self.use_advanced_mt.set(False)
        self.device_preference.set("cuda:0")
        self.log_text.delete(1.0, tk.END)
        self.update_status("就绪")
        
        # Add initial log message back
        self.log_message("界面已重置")
        self._log_device_info()
    
    def check_system_requirements(self):
        """Check system requirements for Windows"""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 9):
            issues.append("Python 版本过低，需要 Python 3.9 或更高版本")
        
        # Check FFmpeg
        try:
            run_kwargs = {"capture_output": True, "text": True, **_subprocess_kwargs()}
            result = subprocess.run(['ffmpeg', '-version'], **run_kwargs)
            if result.returncode != 0:
                issues.append("找不到 FFmpeg 或无法运行")
        except FileNotFoundError:
            issues.append("FFmpeg 未安装或未加入 PATH")
        
        # Check required packages
        try:
            import transformers
        except ImportError as e:
            issues.append(f"缺少必要的 Python 包：{e}")
        
        # Show warnings if any issues found
        if issues:
            warning_text = "检测到以下问题：\n\n" + "\n".join(f"• {issue}" for issue in issues)
            warning_text += "\n\n请运行 install.bat 以解决以上问题。"
            messagebox.showwarning("系统检查", warning_text)
    
    def _log_device_info(self):
        """Log device/GPU information to the log area."""
        self.log_message("设备偏好：CUDA 显卡（cuda:0）")
        if torch.cuda.is_available():
            self.log_message("当前设备：CUDA 显卡")
            self.log_message(f"显卡：{torch.cuda.get_device_name(0)}")
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            self.log_message(f"CUDA 版本：{cuda_version}")
        else:
            self.log_message("当前设备：CPU（回退）")
            self.log_message("提示：如需 GPU 加速，请安装 CUDA 版 PyTorch")

    def show_help(self):
        """Show help dialog with usage instructions"""
        help_text = """
使用说明：

1. 选择视频文件
   - 支持格式：MP4、MKV、AVI、MOV、WMV、FLV
   - 音频越清晰，识别效果通常越好

2. 选择模型与质量参数
   - ASR 默认：Qwen3-ASR-1.7B
   - MT 默认：HY-MT 兼容模型（日语->中文）
   - 质量模式：快速（fast）或 精确（accurate）
   - 音频增强：标准（standard）/ 降噪（denoise）/ 强力降噪（aggressive）
   - 可选 ASR 术语文件：每行一个术语，或使用 `误识别=>标准写法`
   - 分块时长：30-600 秒（默认 120）
   - 重叠时长：默认 2 秒

3. 生成字幕
   - 点击“生成字幕”开始处理
   - 处理时间取决于视频长度、分块设置与模型
   - 生成的 SRT 会保存到输出目录

4. 烧录硬字幕
   - 选择已生成的 SRT 字幕文件
   - 点击“烧录硬字幕”
   - 新视频将包含硬编码中文字幕

注意：
• 首次运行会从 Hugging Face 下载模型，需要联网
• GPU 远快于 CPU，建议安装 CUDA
• 请确保临时文件有足够磁盘空间
• 大文件处理需要耐心等待

排查建议：
• 出错时先检查系统环境与依赖
• 确认 FFmpeg 已正确安装
• 检查网络连接（用于模型下载）
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("帮助")
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
