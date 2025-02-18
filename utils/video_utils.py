import yt_dlp
import subprocess

def get_youtube_stream_url(youtube_url):
    """YouTube 영상의 실제 스트리밍 URL을 가져옴"""
    ydl_opts = {"format": "best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def play_video(playlist, video_index):
    """지정된 YouTube 영상을 FFmpeg를 통해 재생"""
    stream_url = get_youtube_stream_url(playlist[video_index])

    ffmpeg_cmd = [
        "ffmpeg", "-re", "-i", stream_url,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        # "-vf", "scale=640:360",
        # "-b:v", "5000k",   # 비디오 비트레이트 증가 (5000kbps)
        # "-b:a", "192k",    # 오디오 비트레이트 증가
        "-an", "pipe:1"
    ]
    
    return subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
