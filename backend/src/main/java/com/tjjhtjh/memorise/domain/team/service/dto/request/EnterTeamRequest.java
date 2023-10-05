package com.tjjhtjh.memorise.domain.team.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class EnterTeamRequest {
    private Long userSeq;
    private String password;
}
