package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class InviteMemberResponse {
    boolean isSuccess;

    public InviteMemberResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
