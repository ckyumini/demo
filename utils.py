from typing import Tuple, List, Optional, Dict

class HangulDecomposer:
    """한글 자모 분해 및 획순 정보 처리 클래스"""
    
    def __init__(self):
        # 초성/중성/종성 리스트
        self.CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        self.JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
        self.JONGSUNG = [' ','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        
        # 자모별 획수 정보
        self.STROKE_COUNT = {
            'ㄱ': 2, 'ㄴ': 2, 'ㄷ': 3, 'ㄹ': 5, 'ㅁ': 4, 
            'ㅂ': 4, 'ㅅ': 2, 'ㅇ': 1, 'ㅈ': 3, 'ㅊ': 4, 
            'ㅋ': 3, 'ㅌ': 4, 'ㅍ': 4, 'ㅎ': 3, 'ㄲ': 4,
            'ㄸ': 6, 'ㅃ': 8, 'ㅆ': 4, 'ㅉ': 6,
            'ㅏ': 2, 'ㅐ': 3, 'ㅑ': 3, 'ㅒ': 4, 'ㅓ': 2,
            'ㅔ': 3, 'ㅕ': 3, 'ㅖ': 4, 'ㅗ': 2, 'ㅘ': 4,
            'ㅙ': 5, 'ㅚ': 3, 'ㅛ': 3, 'ㅜ': 2, 'ㅝ': 4,
            'ㅞ': 5, 'ㅟ': 3, 'ㅠ': 3, 'ㅡ': 1, 'ㅢ': 2,
            'ㅣ': 1
        }
        
        # 자모별 획순 정보 (상대 좌표)
        self.STROKE_ORDER = self._initialize_stroke_order()
    
    def _initialize_stroke_order(self) -> Dict:
        """자모별 획순 정보 초기화"""
        return {
            # 초성
            'ㄱ': [(0,0,1,0), (1,0,1,1)],                      # 가로선, 세로선
            'ㄴ': [(0,0,0,1), (0,1,1,1)],                      # 세로선, 가로선
            'ㄷ': [(0,0,1,0), (1,0,1,1), (0,1,1,1)],          # 가로선, 세로선, 가로선
            'ㄹ': [(0,0,1,0), (1,0,1,1), (1,1,2,1),           # 가로선, 세로선, 가로선,
                    (2,0,2,1), (2,1,3,1)],                     # 세로선, 가로선
            'ㅁ': [(0,0,0,1), (0,0,1,0), (1,0,1,1),           # 세로선, 가로선, 세로선,
                    (0,1,1,1)],                                # 가로선
            'ㅂ': [(0,0,1,0), (0,0,0,1), (1,0,1,1),           # 가로선, 세로선, 세로선,
                    (0,1,1,1)],                                # 가로선
            'ㅅ': [(0,0,0.5,1), (0.5,1,1,0)],                 # 왼쪽 빗선, 오른쪽 빗선
            'ㅇ': [(0.2,0.2,0.8,0.8)],                        # 원
            'ㅈ': [(0,0,1,0), (0.5,0,0.5,1)],                 # 가로선, 세로선
            'ㅊ': [(0,0,1,0), (0.5,0,0.5,1), (0.2,0.5,0.8,0.5)], # 가로선, 세로선, 가로선
            'ㅋ': [(0,0,1,0), (0.5,0,0.5,1), (0.2,0.5,0.8,0.5)], # 가로선, 세로선, 가로선
            'ㅌ': [(0,0,1,0), (0.5,0,0.5,1), (0.2,0.5,0.8,0.5)], # 가로선, 세로선, 가로선
            'ㅍ': [(0,0,1,0), (0.5,0,0.5,1), (0.2,0.5,0.8,0.5)], # 가로선, 세로선, 가로선
            'ㅎ': [(0.2,0.2,0.8,0.8), (0.5,0.5,0.5,1)],      # 원, 세로선

            # 중성
            'ㅏ': [(0.5,0,0.5,1), (0.5,0.5,1,0.5)],          # 세로선, 가로선
            'ㅐ': [(0.5,0,0.5,1), (0.5,0.5,1,0.5),           # 세로선, 가로선,
                   (0.7,0.3,0.7,0.7)],                        # 점
            'ㅑ': [(0.5,0,0.5,1), (0,0.5,0.5,0.5),           # 세로선, 왼쪽 가로선,
                   (0.5,0.5,1,0.5)],                          # 오른쪽 가로선
            'ㅓ': [(0.5,0,0.5,1), (0,0.5,0.5,0.5)],          # 세로선, 가로선
            'ㅗ': [(0,0.5,1,0.5), (0.5,0.5,0.5,1)],          # 가로선, 세로선
            'ㅜ': [(0,0.5,1,0.5), (0.5,0,0.5,0.5)],          # 가로선, 세로선
            'ㅡ': [(0,0.5,1,0.5)],                           # 가로선
            'ㅣ': [(0.5,0,0.5,1)],                           # 세로선

            # 종성 (초성과 동일한 패턴이지만 크기와 위치가 다름)
            ' ': [],  # 종성 없음
        }
    
    def decompose_hangul(self, char: str) -> Tuple[str, str, str]:
        """한글 문자를 초성/중성/종성으로 분해"""
        if not self.is_hangul(char):
            return None, None, None
            
        char_code = ord(char) - 0xAC00
        jong = char_code % 28
        jung = ((char_code - jong) // 28) % 21
        cho = ((char_code - jong) // 28) // 21
        
        return (
            self.CHOSUNG[cho],
            self.JUNGSUNG[jung],
            self.JONGSUNG[jong] if jong > 0 else None
        )
    
    def get_stroke_order(self, char: str) -> List[List[float]]:
        """문자의 획순 정보 반환"""
        cho, jung, jong = self.decompose_hangul(char)
        strokes = []
        
        # 초성 획순
        if cho:
            strokes.extend(self._adjust_stroke_position(
                self.STROKE_ORDER[cho],
                x_offset=0.0,
                y_offset=0.0
            ))
        
        # 중성 획순
        if jung:
            strokes.extend(self._adjust_stroke_position(
                self.STROKE_ORDER[jung],
                x_offset=0.33,
                y_offset=0.0
            ))
        
        # 종성 획순
        if jong:
            strokes.extend(self._adjust_stroke_position(
                self.STROKE_ORDER[jong],
                x_offset=0.0,
                y_offset=0.66
            ))
        
        return strokes
    
    def get_stroke_count(self, char: str) -> int:
        """문자의 총 획수 반환"""
        cho, jung, jong = self.decompose_hangul(char)
        total = 0
        
        if cho:
            total += self.STROKE_COUNT[cho]
        if jung:
            total += self.STROKE_COUNT[jung]
        if jong:
            total += self.STROKE_COUNT[jong]
            
        return total
    
    def _adjust_stroke_position(self, 
                              strokes: List[List[float]], 
                              x_offset: float, 
                              y_offset: float) -> List[List[float]]:
        """획의 상대 좌표 조정"""
        adjusted = []
        for stroke in strokes:
            new_stroke = []
            for i in range(0, len(stroke), 2):
                new_stroke.extend([
                    stroke[i] + x_offset,
                    stroke[i+1] + y_offset
                ])
            adjusted.append(new_stroke)
        return adjusted
    
    @staticmethod
    def is_hangul(char: str) -> bool:
        """한글 문자 여부 확인"""
        return '\uAC00' <= char <= '\uD7A3'

class DefaultConverter:
    """기본 텍스트 변환기"""
    def __init__(self):
        self.character = list('순국당')  # 기본 문자셋
        self.dict = {char: idx for idx, char in enumerate(self.character)}
        
    def encode(self, text):
        """텍스트를 인덱스로 변환"""
        return [self.dict.get(char, 0) for char in text]
        
    def decode(self, indexes):
        """인덱스를 텍스트로 변환"""
        return ''.join([self.character[idx] if idx < len(self.character) else '' for idx in indexes])

# 전역 인스턴스 생성
hangul_utils = HangulDecomposer()

# 외부 사용을 위한 인터페이스 함수들
def decompose_hangul(char: str) -> Tuple[str, str, str]:
    return hangul_utils.decompose_hangul(char)

def get_stroke_order(char: str) -> List[List[float]]:
    return hangul_utils.get_stroke_order(char)

def get_stroke_count(char: str) -> int:
    return hangul_utils.get_stroke_count(char)

if __name__ == '__main__':
    # 사용 예시
    cho, jung, jong = decompose_hangul('한')
    print(f"초성: {cho}, 중성: {jung}, 종성: {jong}")

    strokes = get_stroke_order('한')
    print(f"획순 정보: {strokes}")

    stroke_count = get_stroke_count('한')
    print(f"총 획수: {stroke_count}")
