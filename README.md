# ssl_cvlab

## update 사항
1. early stopping 기준을 loss로 변경 (120 epoch동안 loss 최저점 이 변동지지 않을 때 학습 조기 종료)
2. config에서 optimizer을 'sgd'로 입력 시 SGD로 변경된 optimizer로 학습 진행 (그 외에는 adam으로 진행)
   
