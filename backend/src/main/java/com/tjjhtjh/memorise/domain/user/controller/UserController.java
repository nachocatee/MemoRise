package com.tjjhtjh.memorise.domain.user.controller;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.UserService;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.request.LoginRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.JoinResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.LoginResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/user")
public class UserController {

    private final UserService userService;

    @PostMapping
    public ResponseEntity<JoinResponse> join(@RequestBody JoinRequest joinRequest) {
        userService.join(joinRequest);
        return ResponseEntity.ok(new JoinResponse(true));
    }
}
