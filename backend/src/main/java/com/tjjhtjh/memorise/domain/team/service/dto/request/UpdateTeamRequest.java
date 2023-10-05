package com.tjjhtjh.memorise.domain.team.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class UpdateTeamRequest {
    private Long userSeq;
    private String name;
    private String password;
}
