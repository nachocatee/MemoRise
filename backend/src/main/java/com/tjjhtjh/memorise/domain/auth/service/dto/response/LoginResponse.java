package com.tjjhtjh.memorise.domain.auth.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class LoginResponse {
    boolean isSuccess;
    String token;

    public LoginResponse(boolean isSuccess, String token) {
        this.isSuccess = isSuccess;
        this.token = token;
    }
}
