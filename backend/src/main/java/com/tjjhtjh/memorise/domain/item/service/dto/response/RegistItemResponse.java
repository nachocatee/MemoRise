package com.tjjhtjh.memorise.domain.item.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class RegistItemResponse {
    boolean isSuccess;

    public RegistItemResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
