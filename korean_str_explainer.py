import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
import os
from datetime import datetime
from utils import decompose_hangul, get_stroke_order  # 상대 경로 제거
from captum.attr import ShapleyValueSampling, Lime
from lime.wrappers.scikit_image import SegmentationAlgorithm

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, converter):
        super().__init__()
        self.model = model
        self.converter = converter
        self.max_length = 25  # 최대 텍스트 길이
        
    def forward(self, x):
        """이미지 입력에 대한 예측 수행"""
        batch_size = x.size(0)
        
        # 빈 텍스트로 초기화된 입력 생성
        length_for_pred = torch.IntTensor([self.max_length] * batch_size).to(x.device)
        text_for_pred = torch.LongTensor(batch_size, self.max_length + 1).fill_(0).to(x.device)
        
        # 예측 수행 (is_train 인자 제거)
        pred = self.model(x, text_for_pred)
        
        return pred

class KoreanSTRExplainer:
    """한글 STR 모델을 위한 설명 생성기"""
    
    def __init__(self, model, converter, device):
        # 로거 설정
        self.setup_logger()
        self.logger.info("KoreanSTRExplainer 초기화 시작")
        
        self.model = ModelWrapper(model, converter).to(device)
        self.converter = converter
        self.device = device
        self.image = None
        self.setup_segmentation()
        self.mask_cache = {}
        self.char_regions_cache = {}
        torch.cuda.empty_cache()
        
        self.logger.info(f"초기화 완료 (디바이스: {device})")
        if torch.cuda.is_available():
            self.logger.info(f"GPU 메모리 상태: {torch.cuda.memory_allocated()/1024**2:.1f}MB 사용 중")

    def setup_segmentation(self):
        """한글 특성을 고려한 세그멘테이션 설정"""
        self.segmentation_fn = SegmentationAlgorithm(
            'quickshift',
            kernel_size=3,  # 한글 획 크기 고려
            max_dist=100,   # 자소 간격 고려
            ratio=0.3,      # 획 연결성 고려
            random_seed=42
        )
        
    def setup_logger(self):
        """로깅 시스템 설정"""
        self.logger = logging.getLogger('KoreanSTRExplainer')
        self.logger.setLevel(logging.INFO)
        
        # 로그 파일 핸들러 설정
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_jamo_attribution(self, char: str, attributions: torch.Tensor) -> Dict:
        """자모 단위 attribution 계산"""
        cho, jung, jong = decompose_hangul(char)
        jamo_attr = {}
        
        if cho:
            cho_mask = self._get_jamo_mask(cho)
            jamo_attr['초성'] = float(torch.mean(attributions[cho_mask]))
            
        if jung:
            jung_mask = self._get_jamo_mask(jung) 
            jamo_attr['중성'] = float(torch.mean(attributions[jung_mask]))
            
        if jong:
            jong_mask = self._get_jamo_mask(jong)
            jamo_attr['종성'] = float(torch.mean(attributions[jong_mask]))
            
        return jamo_attr

    def get_stroke_attribution(self, char: str, attributions: torch.Tensor) -> List[float]:
        """획순 기반 attribution 계산"""
        strokes = get_stroke_order(char)
        stroke_attr = []
        
        for stroke in strokes:
            stroke_mask = self._get_stroke_mask(stroke)
            attr = float(torch.mean(attributions[stroke_mask]))
            stroke_attr.append(attr)
            
        return stroke_attr
        
    def explain_text(self, image: torch.Tensor, text: str) -> Dict:
        """텍스트 전체에 대한 설명 생성"""
        self.logger.info(f"텍스트 '{text}' 설명 시작")
        start_time = datetime.now()
        
        try:
            with torch.cuda.amp.autocast():
                self.image = image.to(self.device)
                self.logger.info(f"이미지 크기: {image.shape}, 디바이스: {self.device}")
                
                explanations = {
                    'global': {},
                    'local': {},
                    'combined': {}
                }
                
                # 전역적 설명
                self.logger.info("전역적 설명 계산 시작")
                global_start = datetime.now()
                global_attr = self._compute_global_attribution(self.image)
                explanations['global'] = self._process_global_attr(global_attr)
                self.logger.info(f"전역적 설명 완료 (소요시간: {(datetime.now()-global_start).total_seconds():.2f}초)")
                
                # 지역적 설명
                self.logger.info("지역적 설명 계산 시작")
                for idx, char in enumerate(text):
                    if '\uAC00' <= char <= '\uD7A3':
                        char_start = datetime.now()
                        try:
                            with torch.no_grad():
                                self.logger.info(f"문자 '{char}' 처리 중...")
                                local_attr = self._compute_local_attribution(self.image, char, idx)
                                jamo_attr = self.get_jamo_attribution(char, local_attr)
                                stroke_attr = self.get_stroke_attribution(char, local_attr)
                                
                            explanations['local'][char] = {
                                'jamo': jamo_attr,
                                'stroke': stroke_attr
                            }
                            self.logger.info(f"문자 '{char}' 처리 완료 (소요시간: {(datetime.now()-char_start).total_seconds():.2f}초)")
                            
                        except Exception as e:
                            self.logger.error(f"문자 '{char}' 처리 중 오류 발생: {str(e)}")
                            continue
                        finally:
                            if torch.cuda.is_available():
                                self.logger.debug(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                                torch.cuda.empty_cache()
                
                # 결합 설명
                self.logger.info("설명 결합 시작")
                explanations['combined'] = self._combine_explanations(
                    explanations['global'],
                    explanations['local']
                )
                
                total_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"설명 생성 완료 (총 소요시간: {total_time:.2f}초)")
                return explanations
                
        except Exception as e:
            self.logger.error(f"설명 생성 중 치명적 오류 발생: {str(e)}")
            raise RuntimeError(f"설명 생성 중 오류 발생: {str(e)}") from e
        finally:
            if torch.cuda.is_available():
                self.logger.info(f"최종 GPU 메모리 상태: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                torch.cuda.empty_cache()
    
    def _compute_global_attribution(self, image: torch.Tensor) -> torch.Tensor:
        """전역적 attribution 계산 (GPU 배치 처리)"""
        batch_size = 32  # 적절한 배치 크기 설정
        svs = ShapleyValueSampling(self.model)
        
        attributions = []
        for i in range(0, image.size(0), batch_size):
            batch = image[i:i+batch_size].to(self.device)
            attr_batch = svs.attribute(batch, target=0)
            attributions.append(attr_batch)
            
        return torch.cat(attributions, dim=0)
    
    def _compute_local_attribution(self, 
                                 image: torch.Tensor,
                                 char: str, 
                                 char_idx: int) -> torch.Tensor:
        """문자별 지역적 attribution 계산"""
        scoring_singlechar = self._get_char_scorer(char_idx)
        svs = ShapleyValueSampling(scoring_singlechar)
        return svs.attribute(image, target=char_idx)
    
    def _combine_explanations(self, 
                            global_attr: Dict,
                            local_attr: Dict) -> Dict:
        """전역적 설명과 지역적 설명 결합"""
        combined = {}
        
        # 가중치 계산
        global_weight = 0.4
        local_weight = 0.6
        
        # 전역-지역 설명 결합
        for char in local_attr:
            combined[char] = {
                'importance': (
                    global_weight * global_attr.get(char, 0) +
                    local_weight * local_attr[char]['jamo']['중성']  # 중성 기준
                ),
                'context': self._get_context_importance(char, local_attr)
            }
            
        return combined
    
    def visualize_explanation(self, 
                            image: torch.Tensor,
                            explanations: Dict,
                            save_path: str):
        """설명 시각화"""
        # 히트맵 생성 로직
        from matplotlib import pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 전역적 설명
        self._plot_heatmap(ax1, image, explanations['global'], "Global")
        
        # 지역적 설명 (자모별)
        self._plot_heatmap(ax2, image, explanations['local'], "Local")
        
        # 결합된 설명
        self._plot_heatmap(ax3, image, explanations['combined'], "Combined")
        
        plt.savefig(save_path)
        plt.close()

    def _plot_heatmap(self, ax, image: torch.Tensor, attribution: Dict, title: str):
        """설명을 히트맵으로 시각화"""
        # 이미지 표시
        ax.imshow(image[0, 0].cpu().numpy(), cmap='gray')
        
        # attribution을 히트맵으로 변환
        heatmap = np.zeros_like(image[0, 0].cpu().numpy())
        
        for char, attr in attribution.items():
            if isinstance(attr, dict):
                if 'importance' in attr:
                    # Combined 설명의 경우
                    value = attr['importance']
                elif 'jamo' in attr:
                    # Local 설명의 경우
                    value = np.mean([v for v in attr['jamo'].values()])
            else:
                # Global 설명의 경우
                value = attr
                
            # 히트맵 업데이트
            char_mask = self._get_char_region(char)
            heatmap[char_mask] = value
        
        # 히트맵 오버레이
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.set_title(title)
        ax.axis('off')

    def _get_jamo_mask(self, jamo: str) -> torch.Tensor:
        """자모 영역에 대한 마스크 생성 (GPU 가속)"""
        if jamo in self.mask_cache:
            return self.mask_cache[jamo]
            
        mask = torch.zeros_like(self.image, device=self.device)  # GPU에 직접 생성
        stroke_info = get_stroke_order(jamo)
        
        if not stroke_info:
            return mask
            
        for stroke in stroke_info:
            x1, y1, x2, y2 = [int(coord * self.image.shape[-1]) for coord in stroke]
            mask[:, :, y1:y2+1, x1:x2+1] = 1.0
            
        self.mask_cache[jamo] = mask
        return mask

    def _get_stroke_mask(self, stroke: List[float]) -> torch.Tensor:
        """획 영역에 대한 마스크 생성 (GPU 가속)"""
        mask = torch.zeros_like(self.image, device=self.device)  # GPU에 직접 생성
        H, W = self.image.shape[-2:]
        
        try:
            x1, y1, x2, y2 = [int(coord * W) for coord in stroke]
            y_coords = torch.arange(y1, y2+1, device=self.device)
            x_coords = torch.arange(x1, x2+1, device=self.device)
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
                mask[:, :, yy, xx] = 1.0
                
        except Exception as e:
            print(f"획 마스크 생성 중 오류 발생: {str(e)}")
            
        return mask

    def _get_context_importance(self, char: str, local_attr: Dict) -> float:
        """주변 문자와의 관계를 고려한 중요도 계산"""
        context_window = 2  # 앞뒤 2글자까지 고려
        char_pos = list(local_attr.keys()).index(char)
        total_chars = len(local_attr)
        
        context_scores = []
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
                
            context_pos = char_pos + offset
            if 0 <= context_pos < total_chars:
                context_char = list(local_attr.keys())[context_pos]
                # 거리에 따른 가중치 적용
                weight = 1.0 / abs(offset)
                context_scores.append(
                    weight * local_attr[context_char]['jamo']['중성']
                )
        
        return sum(context_scores) / len(context_scores) if context_scores else 0.0

    def _process_global_attr(self, attribution: torch.Tensor) -> Dict:
        """전역 attribution 후처리"""
        processed_attr = {}
        
        # 이미지를 문자 영역으로 분할
        char_regions = self._segment_char_regions(attribution)
        
        # 각 문자 영역의 평균 attribution 계산
        for char_idx, region in enumerate(char_regions):
            char = self.converter.decode(char_idx)
            processed_attr[char] = float(torch.mean(attribution[region]))
        
        return processed_attr

    def _segment_char_regions(self, attribution: torch.Tensor) -> List[torch.Tensor]:
        """이미지를 문자 영역으로 분할 (GPU 가속)"""
        # CPU로 이동하여 세그멘테이션 수행
        cpu_attr = attribution.cpu().numpy()
        segments = self.segmentation_fn(cpu_attr)
        char_regions = []
        
        # GPU에서 병렬 처리
        segments_tensor = torch.from_numpy(segments).to(self.device)
        n_chars = len(self.converter.character)
        
        for i in range(min(n_chars, segments.max() + 1)):
            region = (segments_tensor == i)
            char_regions.append(region)
            
        return char_regions
        
    def _get_char_region(self, char: str) -> np.ndarray:
        """문자의 위치에 해당하는 영역 마스크 반환"""
        try:
            if char not in self.char_regions_cache:
                if self.image is None:
                    raise ValueError("이미지가 설정되지 않았습니다")
                    
                segments = self.segmentation_fn(self.image[0, 0].cpu().numpy())
                char_idx = self.converter.encode(char)[0]
                
                if char_idx < segments.max() + 1:
                    self.char_regions_cache[char] = segments == char_idx
                else:
                    self.char_regions_cache[char] = np.zeros_like(
                        self.image[0, 0].cpu().numpy(), 
                        dtype=bool
                    )
                    
            return self.char_regions_cache[char]
            
        except Exception as e:
            print(f"문자 영역 계산 중 오류 발생: {str(e)}")
            return np.zeros_like(self.image[0, 0].cpu().numpy(), dtype=bool)

    def _get_char_scorer(self, char_idx: int):
        """특정 문자에 대한 scorer 생성"""
        class CharScorer(torch.nn.Module):
            def __init__(self, model, char_idx):
                super().__init__()
                self.model = model
                self.char_idx = char_idx
                
            def forward(self, x):
                logits = self.model(x)
                return logits[:, self.char_idx]
        
        return CharScorer(self.model, char_idx)