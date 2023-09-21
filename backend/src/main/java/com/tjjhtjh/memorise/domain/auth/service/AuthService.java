package com.tjjhtjh.memorise.domain.auth.service;

import com.tjjhtjh.memorise.domain.auth.service.dto.request.LoginRequest;
import com.tjjhtjh.memorise.domain.auth.service.dto.response.LoginResponse;

public interface AuthService {

    LoginResponse login(LoginRequest loginRequest);
}
