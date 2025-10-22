
데이터셋 : https://www.kaggle.com/datasets/dhoogla/cicidscollection/data

모델: RandomForest



## Top 20 Features (RandomForest Importance)

- **세로축: 피처 이름 (네트워크 트래픽 특징 값)**
- **가로축: 중요도 (Importance, 0~0.05 정도)**

여기서는 랜덤포레스트가 공격/정상을 구분할 때 가장 많이 참고한 특징이 뭔지 보여줍니다.

상위에 나온 특징들:

- **Fwd Seg Size Min** → 가장 중요한 변수. 공격에서는 패킷 세그먼트 크기 최소값 패턴이 다름.
- **Bwd/Fwd Header Length** → 패킷 헤더 길이. 비정상 트래픽은 헤더 구조가 달라지는 경우 많음.
- **Init Fwd/Bwd Win Bytes** → TCP 윈도우 초기 크기. 봇넷/DoS류에서 비정상적으로 튀는 경우 많음.
- **Packet Length 관련 (Mean, Max, Variance, Std)** → 패킷 길이 분포는 공격 유형마다 고유한 패턴을 보임.
- **Flow Packets/s, Subflow Bytes** → 단위 시간당 패킷 개수/바이트 수. DDoS, Portscan에서 특이하게 커짐.




## Confusion Matrix (혼동 행렬)

- **가로축: 모델이 예측한 값**
- **세로축: 실제 정답**
- 각 칸의 숫자는 “실제로는 X인데, 모델이 Y라고 분류한 개수”를 의미합니다.

사진 속 행렬을 보면:

- `true_Benign`(정상) → 대부분 `pred_Benign`에 모여 있음 → **정상 탐지가 거의 완벽**
- `true_DDoS`, `true_DoS`, `true_Botnet`, `true_Bruteforce` → 대체로 자기 칸에 집중 → **공격 유형도 잘 맞춤**
- `true_Infiltration`, `true_Webattack`, `true_Portscan` → 다른 칸에도 분산 → **희귀 클래스에서 오탐/미탐 존재**

즉, 행렬이 대각선으로 진하게 색칠돼 있으면 잘 맞춘 것이고, 옆으로 퍼지면 혼동이 있다는 뜻.



