package com.tjjhtjh.memorise.domain.auth.controller;

import com.tjjhtjh.memorise.domain.auth.service.AuthService;
import com.tjjhtjh.memorise.domain.auth.service.dto.response.LoginResponse;
import com.tjjhtjh.memorise.domain.user.service.UserService;
import com.tjjhtjh.memorise.global.jwt.service.JwtService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;

@RestController
@RequiredArgsConstructor
@RequestMapping("/auth")
@Slf4j
public class AuthController {

    private final AuthService authService;
    private final UserService userService;
    private final JwtService jwtService;

    @GetMapping("/login/oauth2/code/kakao")
    public ResponseEntity<LoginResponse> login(@RequestParam(required = false) String code) {
        log.info("code : " + code);
        // URL에 포함된 code를 이용하여 액세스 토큰 발급
        String accessToken = authService.getAccessToken(code);
        System.out.println(accessToken);

        // 액세스 토큰을 이용하여 카카오 서버에서 유저 정보(닉네임, 이메일) 받아오기
        HashMap<String, Object> userInfo = authService.getUserInfo(accessToken);
        System.out.println("login Controller : " + userInfo);

        // 만일, DB에 해당 email을 가지는 유저가 없으면 회원가입 시키고 유저 식별자와 JWT 반환
        // 현재 카카오 유저의 전화번호를 받아올 권한이 없어서 테스트를 하지 못함.
        if(userService.getUserInfo(String.valueOf(userInfo.get("email"))) == null) {
            return new ResponseEntity<>(new LoginResponse(false, ""), HttpStatus.OK);
        } else {
            return new ResponseEntity<>(new LoginResponse(true, jwtService.createAccessToken(String.valueOf(userInfo.get("email")))), HttpStatus.OK);
        }
    }

    @GetMapping("/jwt-test")
    public String jwtTest() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        return "jwtTest 요청 성공 : " + authentication.getName();
    }

}