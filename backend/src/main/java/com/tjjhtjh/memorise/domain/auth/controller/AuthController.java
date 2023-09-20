package com.tjjhtjh.memorise.domain.auth.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/auth")
@Slf4j
public class AuthController {

    @GetMapping("/jwt-test")
    public String jwtTest() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        return "jwtTest 요청 성공 : " + authentication.getName();
    }

}