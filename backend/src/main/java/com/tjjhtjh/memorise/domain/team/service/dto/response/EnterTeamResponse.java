package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class EnterTeamResponse {
    boolean isSuccess;

    public EnterTeamResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
