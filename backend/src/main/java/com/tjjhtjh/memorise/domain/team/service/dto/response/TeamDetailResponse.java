package com.tjjhtjh.memorise.domain.team.service.dto.response;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class TeamDetailResponse {
    String name;
    UserInfoResponse me;
    UserInfoResponse owner;
    List<UserInfoResponse> members;
    boolean isPassword;
    String password;

    public TeamDetailResponse(Team team, UserInfoResponse me, UserInfoResponse owner, List<UserInfoResponse> members) {
        this.name = team.getName();
        this.me = me;
        this.owner = owner;
        this.members = members;
        this.isPassword = team.getPassword() != null;
        this.password = team.getPassword();
    }
}
