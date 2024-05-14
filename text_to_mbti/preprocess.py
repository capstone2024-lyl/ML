import pandas as pd

# MBTI CSV 파일 읽기
mbti_df = pd.read_csv('./archive/mbti_1.csv')

# Big Five 데이터 전처리
big_five_df = pd.read_csv('./archive/Big-Five_Backstage.csv')

# 불필요한 컬럼 제거
big_five_df = big_five_df[['text', 'Extraversion', 'Agreeableness', 'Openness', 'Neuroticism', 'Conscientiousness']]

# Big Five 데이터를 MBTI 유형으로 매핑하는 함수
def big_five_to_mbti(e, a, o, n, c):
    # Extraversion -> E/I (외향/내향)
    ei = 'E' if e > 0.5 else 'I'
    # Openness -> S/N (직관/감각)
    sn = 'N' if o > 0.5 else 'S'
    # Agreeableness -> F/T (감정/사고)
    ft = 'F' if a > 0.5 else 'T'
    # Conscientiousness -> J/P (판단/인식)
    jp = 'J' if c > 0.5 else 'P'
    # Neuroticism -> 별도 사용하지 않음

    return f"{ei}{sn}{ft}{jp}"

# Big Five 데이터를 MBTI 유형으로 변환
big_five_df['type'] = big_five_df.apply(
    lambda row: big_five_to_mbti(row['Extraversion'], row['Agreeableness'], row['Openness'], row['Neuroticism'], row['Conscientiousness']),
    axis=1
)

# 컬럼 구조 통일
big_five_df = big_five_df[['type', 'text']].rename(columns={'text': 'posts'})

# MBTI 데이터셋과 Big Five 데이터셋 통합
combined_df = pd.concat([mbti_df, big_five_df], ignore_index=True)

# 통합된 데이터셋 저장
combined_df.to_csv('./archive/combined_mbti_big_five.csv', index=False)

# 통합된 데이터셋 확인
print(combined_df.head())