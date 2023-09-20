package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class LoginResponse {
    boolean isSuccess;

    public LoginResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
