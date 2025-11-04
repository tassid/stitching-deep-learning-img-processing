"""
Video Processor
Processa vídeos e extrai frames para stitching
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from datetime import timedelta


class VideoProcessor:
    """
    Processador de vídeos para extração de frames
    """

    def __init__(self, video_path: str):
        """
        Inicializa o processador de vídeo

        Args:
            video_path: Caminho do arquivo de vídeo
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        # Propriedades do vídeo
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"Vídeo carregado: {self.video_path.name}")
        print(f"  - Resolução: {self.width}x{self.height}")
        print(f"  - FPS: {self.fps:.2f}")
        print(f"  - Total de frames: {self.total_frames}")
        print(f"  - Duração: {self._format_duration(self.duration)}")

    def __del__(self):
        """Release do vídeo ao destruir objeto"""
        if hasattr(self, 'cap'):
            self.cap.release()

    def _format_duration(self, seconds: float) -> str:
        """Formata duração em formato legível"""
        return str(timedelta(seconds=int(seconds)))

    def extract_frames_by_interval(self, interval: int = 10,
                                   max_frames: Optional[int] = None,
                                   start_frame: int = 0) -> List[np.ndarray]:
        """
        Extrai frames em intervalos regulares

        Args:
            interval: Intervalo entre frames (ex: 10 = a cada 10 frames)
            max_frames: Número máximo de frames a extrair (None = todos)
            start_frame: Frame inicial

        Returns:
            Lista de frames extraídos
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame
        extracted = 0

        print(f"\nExtraindo frames a cada {interval} frames...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if (frame_count - start_frame) % interval == 0:
                frames.append(frame.copy())
                extracted += 1

                if extracted % 10 == 0:
                    print(f"  Extraídos {extracted} frames...")

                if max_frames and extracted >= max_frames:
                    break

            frame_count += 1

        print(f"✓ Total de {len(frames)} frames extraídos")
        return frames

    def extract_frames_by_time(self, time_interval: float = 1.0,
                               max_frames: Optional[int] = None,
                               start_time: float = 0.0) -> List[np.ndarray]:
        """
        Extrai frames em intervalos de tempo

        Args:
            time_interval: Intervalo em segundos (ex: 1.0 = 1 frame por segundo)
            max_frames: Número máximo de frames
            start_time: Tempo inicial em segundos

        Returns:
            Lista de frames extraídos
        """
        # Converter tempo para frames
        frame_interval = int(time_interval * self.fps)
        start_frame = int(start_time * self.fps)

        return self.extract_frames_by_interval(
            interval=frame_interval,
            max_frames=max_frames,
            start_frame=start_frame
        )

    def extract_key_frames(self, threshold: float = 30.0,
                          min_interval: int = 10,
                          max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extrai keyframes baseado em diferença entre frames

        Args:
            threshold: Limiar de diferença (maior = mais seletivo)
            min_interval: Intervalo mínimo entre keyframes
            max_frames: Número máximo de frames

        Returns:
            Lista de keyframes
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, prev_frame = self.cap.read()
        if not ret:
            return frames

        frames.append(prev_frame.copy())
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 1
        last_keyframe = 0

        print(f"\nExtraindo keyframes (threshold={threshold})...")

        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break

            # Verificar intervalo mínimo
            if frame_count - last_keyframe < min_interval:
                frame_count += 1
                continue

            # Calcular diferença
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = np.mean(diff)

            # Se diferença for significativa, é um keyframe
            if mean_diff > threshold:
                frames.append(curr_frame.copy())
                prev_gray = curr_gray
                last_keyframe = frame_count

                if len(frames) % 10 == 0:
                    print(f"  Extraídos {len(frames)} keyframes...")

                if max_frames and len(frames) >= max_frames:
                    break

            frame_count += 1

        print(f"✓ Total de {len(frames)} keyframes extraídos")
        return frames

    def extract_and_save_frames(self, output_dir: str,
                               interval: int = 10,
                               prefix: str = "frame",
                               format: str = "jpg") -> List[str]:
        """
        Extrai e salva frames em disco

        Args:
            output_dir: Diretório de saída
            interval: Intervalo entre frames
            prefix: Prefixo dos arquivos
            format: Formato de saída (jpg, png)

        Returns:
            Lista de caminhos dos arquivos salvos
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frames = self.extract_frames_by_interval(interval)
        saved_paths = []

        print(f"\nSalvando {len(frames)} frames em {output_dir}...")

        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:04d}.{format}"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(str(filepath))

            if (i + 1) % 10 == 0:
                print(f"  Salvos {i + 1}/{len(frames)} frames...")

        print(f"✓ Frames salvos com sucesso")
        return saved_paths

    def process_with_callback(self, callback: Callable[[np.ndarray, int], any],
                             interval: int = 1,
                             max_frames: Optional[int] = None) -> List[any]:
        """
        Processa frames com função callback

        Args:
            callback: Função callback(frame, frame_number) -> result
            interval: Intervalo entre frames
            max_frames: Número máximo de frames

        Returns:
            Lista de resultados do callback
        """
        results = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0
        processed = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                result = callback(frame, frame_count)
                results.append(result)
                processed += 1

                if max_frames and processed >= max_frames:
                    break

            frame_count += 1

        return results

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Obtém frame específico

        Args:
            frame_number: Número do frame

        Returns:
            Frame ou None se não encontrado
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        return frame if ret else None

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Obtém frame em tempo específico

        Args:
            time_seconds: Tempo em segundos

        Returns:
            Frame ou None se não encontrado
        """
        frame_number = int(time_seconds * self.fps)
        return self.get_frame_at(frame_number)


def extract_frames(video_path: str,
                  interval: int = 10,
                  max_frames: Optional[int] = None,
                  method: str = 'interval') -> List[np.ndarray]:
    """
    Função auxiliar para extrair frames de vídeo

    Args:
        video_path: Caminho do vídeo
        interval: Intervalo (frames ou segundos dependendo do método)
        max_frames: Número máximo de frames
        method: 'interval', 'time', ou 'keyframes'

    Returns:
        Lista de frames
    """
    processor = VideoProcessor(video_path)

    if method == 'interval':
        return processor.extract_frames_by_interval(interval, max_frames)
    elif method == 'time':
        return processor.extract_frames_by_time(interval, max_frames)
    elif method == 'keyframes':
        return processor.extract_key_frames(max_frames=max_frames)
    else:
        raise ValueError(f"Método inválido: {method}")
