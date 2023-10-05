package com.tjjhtjh.memorise.domain.team.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class CreateTeamRequest {
    private String name;
    private Long owner;
    private String password;
}
