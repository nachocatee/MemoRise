package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class CreateTeamResponse {
    boolean isSuccess;

    public CreateTeamResponse(boolean isSuccess) {
        this.isSuccess = isSuccess;
    }

}
