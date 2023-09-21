package com.tjjhtjh.memorise.domain.team.service.dto.request;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class TeamRequest {
    private String name;
    private Long owner;
    private String password;
    private User user;
    private String userId;

    public Team toRegistEntity(User user, String groupName) {
        Team team = Team.builder()
                .name(groupName)
                .owner(user.getUserSeq())
                .password(password)
                .build();
        return team;
    }
}
