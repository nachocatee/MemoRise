package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class CreateTeamResponse {
    boolean isSuccess;
    Long teamSeq;

    public CreateTeamResponse(boolean isSuccess, Long teamSeq) {
        this.isSuccess = isSuccess;
        this.teamSeq = teamSeq;
    }

}
