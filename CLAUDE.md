# codeconvention
클로드가 켜지면 제일먼저 https://github.com/Thomas-Anderson1999/ThalamusAgentConvention 의 내용이 서브모듈에 들어있는지, 내용이 최신버전인지 확인하고, 최신버전에 맞춰 업데이트할것

# Project 
@.conventions/thalamus/CLAUDE.md
- Project Name:ThalamusROS2_Controller
- Python: 3.12


## 디렉토리, 파일
venv: ThalamusVisualRecognition 프로젝트 전용 venv
requirements.txt : comm_venv312에 pip으로 설치한 것이 있으면 여기에 기록해서 차후에 한번에 인스톨 할수 있게 할것, 이미 설치된것도 차후에 복구하기 위해 기록할것


### Ruff 검사
커밋 전 Ruff 검사는 서브모듈(ThalamusEngine, .conventions)을 제외하고 실행한다.
```bash
ruff format --check --config .conventions/thalamus/pyproject.toml --exclude ThalamusEngine --exclude .conventions .
ruff check --config .conventions/thalamus/pyproject.toml --exclude ThalamusEngine --exclude .conventions .
```

## 하부 프로젝트
각자 하나의 탭을 가진다.
프로젝트 파일: visual_recognition.py이 하부프로젝트의 파이썬 프로젝트를 부르는 구조로 되어있다. ui에서는 되도록 간결하게 부르기만하고, 실제 연산은 하부/실제연산.py가 하게한다
(1) ui_controller : ui로 조작하는 로버 
(2) serial_bypassing: 커맨드 바이패스 해주는 것 (보드에서 실행)
(3) camera_node_d455i : 이미지 전송 (보드에서 실행)

## README.md내용
1.리포지토리를 새로 받았을때 복구할수 있는 설치방법을 설명한다.
2. 하부프로젝트의 내용을 간략하게 소개한다. 

## 프로젝트 진행
각 프로젝트에는 todo.md와 done.md를 갖는다. todo.md에 해야할 일이 적혀있으면, done은 구현된 내용을 적는다. todo.md는 무엇을 해야하고, 어떤 걸로 완료했는지를 판단하는 항목이 기술되어 있어야하고 done.md는 commitLog.txt에서 특정할수 있는 내용이 있어야한다.
todo에는 story단위에 task(단계)를 적어놓고, 스토리가 완료되지 않으면 todo의 내용에는 완료된 내용에 대하여 체크로 남겨둔다. 스토리가 완료되면, 간결하게 내용을 정리하여 done.md에 스토리,task단위로 저장한다. 

## document
알고리즘의 내용을 기록한다.




