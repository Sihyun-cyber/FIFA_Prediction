import pandas as pd

# 확인할 선수 이름을 여기에 입력하세요.
player_to_check = 'Son Heung-Min'

try:
    df = pd.read_csv('combined_fifa_data_processed.csv')

    # 해당 선수의 데이터를 모두 찾습니다.
    player_data = df[df['player_name'] == player_to_check].copy()

    if not player_data.empty:
        print(f"'{player_to_check}' 선수의 연도별 player_id:")
        
        # 'year'와 'player_id' 컬럼만 선택해서 보여줍니다.
        # 연도순으로 정렬합니다.
        player_info = player_data[['year', 'player_id']].sort_values(by='year').reset_index(drop=True)
        print(player_info)

    else:
        print(f"'{player_to_check}' 선수를 찾을 수 없습니다.")

except FileNotFoundError:
    print("오류: combined_fifa_data_processed.csv 파일을 찾을 수 없습니다.")