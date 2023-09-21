package com.tjjhtjh.memorise.domain.team.service.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
public class TeamResponse {

    private String name;
    private Long owner;
    private String password;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer ownerCheck;

}
