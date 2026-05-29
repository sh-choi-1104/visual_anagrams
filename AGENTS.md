### Requirements
Latent Diffusion Model인 SDXL기반으로 hybrid image를 만든다. 여기서 정의하는 hybrid image란, 멀리서 봤을 때는 low frequency가 부각되어 보이고, 가까이서 봤을 때는 high frequeny가 부각되는 이미지로, 보는 거리에 따라 다르게 보이는 이미지이다.

### Methods
1. 강화학습 방법론인 DRTune을 활용한다. close_prompt와 far_prompt를 별도로 두어 step에 따라 비율을 조절하는 강화방식을 적용한다.
2. Denoising Inference 시에 적절한 Scheduling 을 통해 이미지를 생성한다. Diffusion model의 경우 Early Step에는 Low frequency 정보를 생성하고 Late step에는 high frequency 정보를 생성한다는 사전지식이 있으므로, Denoising scheduling 시에도 이 방법을 적극 고려한다. 즉, Denoising step에 따라 두가지 text condition의 비율을 잘 조절한다.

### Guidelines
1. Reference code인 visual anagram 코드를 참고한다.
2. Latent 기반 변형을 위해 작성된 코드와 RL 관련 코드를 참고한다.
3. 코드를 생성하면서 반복적으로 필요할 것이라고 판단되는 부분은 이 문서의 Harness에 추가하여 작업을 효율화 한다.
4. 불필요하거나 비효율적인 코드가 있다면 해당 부분을 주기적으로 개선한다.
5. Hybrid Image가 잘 생성될 만한 hparam 비율을 알아서 잘 조정하고, 실험세팅을 바꿔가며 실험해본다.
6. 기본적으로 LoRA 학습으로 진행하되, single gpu에서 진행하므로 rank를 가능한한 키운다.
7. 각 실험에 대한 내용은 wandb로 잘 비교할 수 있도록 잘 정리한다.

### Harness
1. 테스트용으로 test/smoke run등을 했을 경우에는 결과 확인후 해당 파일을 바로 정리한다. 바로 정리가 어렵다고 판단되면, 그대로 남겨두되 주기적으로 정리한다.
