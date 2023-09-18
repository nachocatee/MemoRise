package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class JoinResponse {
    boolean isSuccess;

    public JoinResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
