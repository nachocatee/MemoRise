package com.tjjhtjh.memorise.domain.team.service.dto.response;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class MyTeamListResponse {
    private Long teamSeq;
    private String teamName;
    private String myProfile;
    private boolean isOwner;
    private String ownerProfile;
    private List<String> memberProfiles;

    public MyTeamListResponse(Team team, User user, List<String> memberProfiles) {
        this.teamSeq = team.getTeamSeq();
        this.teamName = team.getName();
        this.myProfile = user.getProfile();
        this.isOwner = team.getOwner().equals(user.getUserSeq());
        this.ownerProfile = isOwner ? null : memberProfiles.get(0);
        this.memberProfiles = isOwner ? memberProfiles : memberProfiles.subList(1, memberProfiles.size());
    }
}
