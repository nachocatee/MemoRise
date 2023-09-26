package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class ExitTeamResponse {
    boolean isSuccess;

    public ExitTeamResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
