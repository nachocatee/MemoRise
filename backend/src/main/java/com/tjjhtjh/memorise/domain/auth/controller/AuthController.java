package com.tjjhtjh.memorise.domain.auth.controller;

import com.tjjhtjh.memorise.domain.auth.service.AuthService;
import com.tjjhtjh.memorise.domain.auth.service.dto.request.LoginRequest;
import com.tjjhtjh.memorise.domain.auth.service.dto.response.LoginResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/auth")
public class AuthController {

    private final AuthService authService;

    @PostMapping
    public ResponseEntity<LoginResponse> login(@RequestBody LoginRequest loginRequest) {
        return ResponseEntity.ok(authService.login(loginRequest));
    }
}
