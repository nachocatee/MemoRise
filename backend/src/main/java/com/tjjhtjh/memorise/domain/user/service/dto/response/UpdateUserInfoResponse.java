package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class UpdateUserInfoResponse {
    boolean isSuccess;

    public UpdateUserInfoResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
