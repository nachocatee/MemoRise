# [B106] Memorise

Memorise 프로젝트는 Docker, Docker Compose, 그리고 Jenkins Pipeline을 활용한 CI/CD 자동화 환경을 구성하고 있습니다.

Gitlab의 Webhook 설정으로 인해 Push 또는 Merge 이벤트 발생 시, Jenkins Pipeline을 통해 자동 빌드와 배포가 이루어집니다.

프론트엔드 부분은 apk 환경에서 빌드하고 배포합니다. 

백엔드는 Gradle을 사용하여 빌드하며, Docker를 통해 컨테이너를 관리하고 배포합니다.


## Version

### Frontend

| Type             | Version |
| ---------------- | ------- |
| JavaScript       | ES6     |
| React-Native     | 0.72.4  |
| Redux            | 8.1.2   |
| npm              | 9.5.1   |
| Axios            | 1.5.0   |
| WebRTC           | 111.0.3 |

### Backend

| Type              | Version          |
| ----------------- | ------------     |
| Java              | openjdk : 17.0.3 |
| Spring Boot       | 3.1.3            |
| Gradle            | 8.2.1            |
| Spring Security   | 6.1.3            |
| JPA               | -                |
| Hibernate         | -                |
| QueryDsl          | 5.0.0            |

### AI

| Type       | Version   |
| ---------- | --------- |
| Anaconda   | 23.7.2    |
| Python     | 3.9.16    |


### Database

| Type            | Version |
| --------------- | ------- |
| MySQL           | 8.0.33  |
| MongoDB         | 4.4.24  |


## Nginx Port forwarding

| Port   | Content     |
| ------ | ----------- |
| 80     | HTTP        |
| 3306   | MySQL       |
| 8000   | API Gateway |
| 9000   | Jenkins     |
| 9090   | Prometheus  |
| 3000   | Grafana     |
| 27017  | MongoDB     |


## Docker And Docker Compose Install

https://docs.docker.com/engine/install/ubuntu/


## Environment Variables

### Backend

- application.yml 내부에서 ${ENV} 처럼 환경변수를 설정하여 GIT에 yml파일을 업로드 하였고,
환경변수는 Dockerfile 내부에서 따로 관리하여주었습니다.

- mysql 172.26.6.59 IP 주소로 외부접속 허용 (ID/PW 로그인 후 사용 가능)
- mongoDB 172.26.6.59 IP 주소로 외부접속 허용 (ID/PW 로그인 후 사용 가능)

```sh
# application.yml

spring:
  main:
    allow-bean-definition-overriding: true
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: ${DB_URL}
    username: ${DB_USERNAME}
    password: ${DB_PASSWORD}
  jpa:
    hibernate:
      ddl-auto: none
    properties:
      hibernate:
        default_batch_fetch_size: 500
  servlet:
    multipart:
      max-file-size: 20MB
      max-request-size: 50MB
  jwt:
    prefix: 'Bearer '
    secret: 2BBE0C48B91A7D1B8A6753A8B9CBE1DB16B84379F3F91FE115621284DF7A48F1CD71E9BEB90EA614C7BD924250AA9E446A866725E685A65DF5D139A5CD180DC9 # test -> sha512
    token:
      access-expiration-time: ${ACCESS_EXPIRATION}
      test-access-expiration-time: 5000
      refresh-expiration-time: ${REFRESH_EXPIRATION}
  security:
    oauth2:
      client:
        provider:
          kakao:
            authorization-uri: https://kauth.kakao.com/oauth/authorize
            token-uri: https://kauth.kakao.com/oauth/token
            user-info-uri: https://kapi.kakao.com/v2/user/me
            user-name-attribute: id
        registration:
          kakao:
            client-id: ${CLIENT_ID}
            client-secret: ${CLIENT_SECRET}
            client-authentication-method: client_secret_post
            redirect-uri: http://localhost:8000/auth/oauth2/code/kakao
            authorization-grant-type: authorization_code
            client-name: kakao
            scope:
              - account_email
server:
  port: 8000

management:
  endpoints:
    web:
      exposure:
        include: health,info,prometheus
  endpoint:
    metrics:
      enabled: true
    prometheus:
      enabled: true
  prometheus:
    metrics:
      export:
        enabled: true
cloud:
  aws:
    s3:
      bucket: ${AWS_S3_BUCKET}
    credentials:
      access-key: ${AWS_ACCESS_KEY}
      secret-key: ${AWS_SECRET_KEY}
    region:
      static: ap-northeast-2
      auto: false
    stack:
      auto: false
```

### Database ERD

link : [ERDCLOUD LINK](https://steady-volcano-b48.notion.site/ERD-f830550e31ca4aa984efb8c5fe98e65c?pvs=4)

## Docker : Dockerfile

Dockerfile은 git에 작성하지 않고 ubuntu에서 vim으로 작성 후 사용하였습니다.

### Backend

```sh

FROM jenkins/jenkins:jdk17
  
COPY backend/build/libs/memorise-0.0.1-SNAPSHOT.jar /app.jar

ENV DB_URL 
ENV DB_USERNAME tjjh
ENV DB_PASSWORD 
ENV CLIENT_ID 
ENV CLIENT_SECRET 
ENV ACCESS_EXPIRATION 
ENV REFRESH_EXPIRATION 
ENV AWS_S3_BUCKET 
ENV AWS_ACCESS_KEY 
ENV AWS_SECRET_KEY 


EXPOSE 8000

CMD ["nohup", "java", "-jar", "app.jar", ">/dev/null", "2>&1", "&"]

```


## Docker Compose : docker-compose.yml

```sh
version: '3'

services:
    jenkins:
        image: jenkins/jenkins:jdk17
        container_name: jenkins
        environment:
            - TZ=Asia/Seoul
        volumes:
            - /var/run/docker.sock:/var/run/docker.sock
            - /jenkins:/var/jenkins_home
            - /usr/bin/docker:/usr/bin/docker
        ports:
            - "9000:8080"
        user: root

```


## Jenkins

- jenkins admin ID : memorise-b106-teuk
- jenkins admin password : wlrmatlrks3@tl@40qns@dhgn

### Jenkins Job

- BE-pipeline 

### Jenkins file

Jenkins file git에 작성하지 않고, Jenkins 내부에 pipeline을 직접 작성하였다.

### Backend

- docker container name : teuk
- docker image name : memo

```sh
pipeline{
    agent any
    
    stages {
        stage('Git Clone') {
            steps {
                
                git branch: 'back', credentialsId: '74859fb2-f120-471b-a1a7-baa548bd8a75', url: 'https://lab.ssafy.com/s09-ai-image-sub2/S09P22B106.git'
                
            }
        }
        stage('BE-Build') {
            steps {
                dir("./backend") {
                    sh 'chmod +x gradlew'
                    sh  './gradlew clean build'

                }
            }
        }
        
        stage('Dockerize') {
            steps {
						
                sh '''
                    docker stop teuk || true  
                    docker rm teuk || true
                    docker rmi memo || true
                    docker build -t memo .
                '''
								
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker run -d --name teuk -p 8000:8000 memo'
								
            }
        }
    }
}
```
