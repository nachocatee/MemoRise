package com.tjjhtjh.memorise.domain.auth.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class LoginResponse {
    boolean isSuccess;
    Long userSeq;

    public LoginResponse(boolean isSuccess, Long userSeq) {
        this.isSuccess = isSuccess;
        this.userSeq = userSeq;
    }
}
