import torch
import easyocr
from korean_str_explainer import KoreanSTRExplainer
import cv2
import numpy as np
from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_korean_str_explainer():
    try:
        # CUDA 설정 및 확인
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {device}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
            print(f"가용 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        print("\n=== 모델 초기화 시작 ===")
        # EasyOCR 설정
        reader = easyocr.Reader(['ko'], gpu=True)
        print("EasyOCR 리더 초기화 완료")
        
        recognizer = reader.recognizer
        print("인식기 추출 완료")
        
        # DataParallel 모델에서 실제 모델 추출
        if isinstance(recognizer, torch.nn.DataParallel):
            base_model = recognizer.module
        else:
            base_model = recognizer
        print("기본 모델 추출 완료")
        
        # converter는 reader에서 직접 가져옴
        try:
            converter = reader.converter
            print("Reader에서 converter 추출 완료")
        except AttributeError:
            try:
                converter = reader.recognition.converter
                print("Recognition 모듈에서 converter 추출 완료")
            except AttributeError:
                print("Warning: converter를 찾을 수 없습니다. 기본 converter를 사용합니다.")
                from utils import DefaultConverter  # DefaultConverter 구현 필요
                converter = DefaultConverter()
        
        # 모델을 평가 모드로 설정
        base_model.eval()
        model = base_model.to(device)
        print(f"모델 평가 모드 설정 및 {device}로 이동 완료")
        
        # 디렉토리 확인 및 생성
        os.makedirs("test_images", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        print("\n=== 이미지 처리 시작 ===")
        # 테스트 이미지 로드 및 전처리
        img_path = os.path.join("test_images", "test.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {img_path}")
        print(f"이미지 경로 확인: {img_path}")
            
        img = Image.open(img_path).convert('L')
        img = img.resize((224, 224))
        print("이미지 로드 및 크기 조정 완료")
        
        # 이미지를 텐서로 변환
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        print(f"이미지 텐서 변환 및 {device}로 이동 완료")
        
        print("\n=== 설명 생성 시작 ===")
        # KoreanSTRExplainer 초기화 및 실행
        explainer = KoreanSTRExplainer(model, converter, device)
        print("설명 생성기 초기화 완료")
        
        explanations = explainer.explain_text(img_tensor, "순국당")
        print("설명 생성 완료")
        
        print("\n=== 결과 저장 시작 ===")
        # 결과 저장
        save_path = os.path.join("results", "순국당_설명.png")
        explainer.visualize_explanation(img_tensor, explanations, save_path)
        print(f"결과 이미지 저장 완료: {save_path}")
        
        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            print(f"\n현재 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
            print(f"최대 GPU 메모리 사용량: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f}MB")
            print(f"캐시된 GPU 메모리: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB")
        
        # 결과 검증
        print("\n=== 결과 검증 ===")
        assert 'global' in explanations, "전역 설명 누락"
        assert 'local' in explanations, "지역 설명 누락"
        assert 'combined' in explanations, "결합 설명 누락"
        
        print("\n테스트 완료!")
        print(f"결과 이미지가 {save_path}에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n테스트 실행 중 오류 발생: {str(e)}")
        if torch.cuda.is_available():
            print(f"오류 발생 시점의 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
        raise

if __name__ == '__main__':
    test_korean_str_explainer()