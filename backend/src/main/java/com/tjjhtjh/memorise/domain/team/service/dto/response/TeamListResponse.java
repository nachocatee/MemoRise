package com.tjjhtjh.memorise.domain.team.service.dto.response;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Getter
@NoArgsConstructor
public class TeamListResponse {

    private Long teamSeq;
    private String teamName;
    private List<String> profiles;
    private boolean isPassword;

    public TeamListResponse(Team team, List<String> profiles) {
        this.teamSeq = team.getTeamSeq();
        this.teamName = team.getName();
        this.profiles = profiles;
        this.isPassword = team.getPassword() != null;
    }
}
