package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class UpdateTeamResponse {
    boolean isSuccess;

    public UpdateTeamResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }
}
