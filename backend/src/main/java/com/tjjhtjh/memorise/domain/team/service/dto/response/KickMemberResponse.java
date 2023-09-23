package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class KickMemberResponse {
    private boolean isSuccess;

    public KickMemberResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
