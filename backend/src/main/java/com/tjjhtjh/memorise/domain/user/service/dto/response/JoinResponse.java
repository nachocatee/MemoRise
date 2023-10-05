package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class JoinResponse {
    boolean isSuccess;
    Long userSeq;

    public JoinResponse(boolean isSuccess, Long userSeq) {
        this.isSuccess = isSuccess;
        this.userSeq = userSeq;
    }

}
