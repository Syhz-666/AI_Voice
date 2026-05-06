import gradio as gr
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

from module_utils import AudioCNN, preprocess_audio


plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 从 numpy 音频数据直接提取 MFCC（避免写临时文件）
def preprocess_audio_from_array(y, sr=16000, target_seconds=7, n_mfcc=40, n_fft=1024, hop_length=512):
    target_length = sr * target_seconds
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mfcc

# -------------------- 加载模型（应用启动时执行） --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioCNN().to(device)

# 找任意一个 wav 文件来触发动态全连接层创建
dummy_path = r"C:\Users\admin\Desktop\work\chuangxinshijian\forth\test.wav"   # 改成你电脑上真实存在的一个 wav 路径
dummy_input = preprocess_audio(dummy_path).to(device)
with torch.no_grad():
    _ = model(dummy_input)

model.load_state_dict(torch.load('audio_classifier/audio_classifier2.pth', map_location=device))
model.eval()
print("模型加载完成！")

# -------------------- 生成波形图 --------------------
def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color='steelblue', linewidth=0.8)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title("波形图")
    plt.tight_layout()
    return fig

# -------------------- 生成梅尔频谱图 --------------------
def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title("梅尔频谱图")
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# -------------------- 推理函数 --------------------
def predict_wav(audio_file):
    # 预处理并推理
    input_tensor = preprocess_audio(audio_file).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        prob = torch.softmax(outputs, dim=1)[0, 1].item()

    # 结果标签（用于 gr.Label）
    result_dict = {
        "真实人声": 1 - prob,
        "AI合成声": prob
    }

    # 生成图表
    wave_fig = plot_waveform(audio_file)
    mel_fig = plot_mel_spectrogram(audio_file)

    # 颜色条：AI概率越高越红，低则绿
    color_html = f"""
    <div style="width:100%; height:25px; background: linear-gradient(to right, 
        green 0%, yellow 50%, red 100%); border-radius:5px; position:relative;">
        <div style="position:absolute; left:{prob*100}%; top:-2px; width:4px; height:29px; 
            background:black; border-radius:2px;"></div>
    </div>
    <p style="font-weight:bold; text-align:center;">AI合成概率：{prob*100:.2f}%</p>
    """

    return result_dict, wave_fig, mel_fig, audio_file, color_html

# 基于音频数组绘制波形图
def plot_waveform_from_array(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color='steelblue', linewidth=0.8)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title("波形图（实时）")
    plt.tight_layout()
    return fig

# 基于音频数组绘制梅尔频谱图
def plot_mel_from_array(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title("梅尔频谱图（实时）")
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# 空白占位图（没有音频时用）
def plot_waveform_empty():
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title("波形图")
    plt.tight_layout()
    return fig

def plot_mel_empty():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("梅尔频谱图")
    plt.tight_layout()
    return fig

# -------------------- Gradio 界面 --------------------
with gr.Blocks(title="音频真伪鉴别") as demo:
    gr.Markdown("## ️ 音频真伪鉴别：真实人声 vs AI合成声")

    with gr.Tabs():
        # ---------- 标签页1：上传文件检测 ----------
        with gr.TabItem(" 上传文件检测"):
            gr.Markdown("上传一段 .wav 语音文件，系统将判断是否为 AI 合成。")
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="上传语音文件 (.wav)")
                    play_output = gr.Audio(type="filepath", label="播放音频")
                with gr.Column():
                    label_output = gr.Label(label="检测结果")
                    color_output = gr.HTML()
            with gr.Row():
                wave_plot = gr.Plot(label="波形图")
                mel_plot = gr.Plot(label="梅尔频谱图")
            btn_upload = gr.Button("开始检测", variant="primary")
            btn_upload.click(
                fn=predict_wav,
                inputs=audio_input,
                outputs=[label_output, wave_plot, mel_plot, play_output, color_output]
            )

        # ---------- 标签页2：实时录音检测（流式） ----------
        with gr.TabItem(" 实时录音检测"):
            gr.Markdown("点击麦克风按钮开始说话，系统会实时分析您的声音。建议说够 3 秒以上以得到较准结果。")

            with gr.Row():
                with gr.Column():
                    mic_input = gr.Audio(sources=["microphone"], type="numpy", streaming=True,
                                         label="实时录音")
                with gr.Column():
                    mic_label = gr.Label(label="实时检测结果")
                    mic_color = gr.HTML()
            with gr.Row():
                mic_wave = gr.Plot(label="波形图", every=1)
                mic_mel = gr.Plot(label="梅尔频谱图", every=1)

            # 流式处理函数（内部累积音频，不使用 gr.State）
            def streaming_predict(audio_chunk):
                accumulated = np.array([], dtype=np.float32)

                target_sr = 16000

                while True:
                    # 兼容 Gradio 可能传入 (sr, array) 元组
                    if isinstance(audio_chunk, tuple):
                        audio_chunk = audio_chunk[1]
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        accumulated = np.concatenate([accumulated, audio_chunk.flatten()])

                    sr = 16000
                    target_samples = sr * 7
                    recent = accumulated[-target_samples:] if len(accumulated) > target_samples else accumulated

                    label_output = {"等待足够语音...": 1.0}
                    color_html = "<div style='color:gray;'>请持续说话，系统正在监听...</div>"
                    wave_fig = plot_waveform_empty()
                    mel_fig = plot_mel_empty()

                    if len(recent) >= sr * 1:
                        try:
                            input_tensor = preprocess_audio_from_array(recent, sr=sr).to(device)
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                prob = torch.softmax(outputs, dim=1)[0, 1].item()
                            label_output = {"真实人声": 1 - prob, "AI合成声": prob}
                            color_html = f"""
                            <div style="width:100%; height:25px; background: linear-gradient(to right, 
                                green 0%, yellow 50%, red 100%); border-radius:5px; position:relative;">
                                <div style="position:absolute; left:{prob * 100}%; top:-2px; width:4px; height:29px; 
                                    background:black; border-radius:2px;"></div>
                            </div>
                            <p style="font-weight:bold; text-align:center;">AI合成概率：{prob * 100:.2f}%</p>
                            """
                            wave_fig = plot_waveform_from_array(recent, sr)
                            mel_fig = plot_mel_from_array(recent, sr)
                        except Exception as e:
                            label_output = {"错误": 1.0}
                            color_html = f"<div style='color:red;'>推理出错: {str(e)}</div>"
                    else:
                        if len(recent) > 0:
                            wave_fig = plot_waveform_from_array(recent, sr)
                            mel_fig = plot_mel_from_array(recent, sr)

                    # 产出当前结果，等待下一个音频块
                    audio_chunk = yield label_output, color_html, wave_fig, mel_fig

            # 绑定流事件（不再需要 gr.State）
            mic_input.stream(
                fn=streaming_predict,
                inputs=[mic_input],
                outputs=[mic_label, mic_color, mic_wave, mic_mel],
                show_progress=False
            )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

