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
    private String myProfile;
    private boolean isParticipated;
    private boolean isOwner;
    private String ownerProfile;
    private List<String> memberProfiles;
    private boolean isPassword;

    public TeamListResponse(Team team, User user, String ownerProfile, List<String> memberProfiles, boolean isParticipated) {
        this.teamSeq = team.getTeamSeq();
        this.teamName = team.getName();
        this.myProfile = user != null ? user.getProfile() : null;
        this.isParticipated = isParticipated;
        this.isOwner = team.getOwner().equals(user.getUserSeq());
        this.ownerProfile = isOwner ? null : ownerProfile;
        this.memberProfiles = isParticipated ? (isOwner ? memberProfiles.subList(0, memberProfiles.size()) : memberProfiles.subList(0, Math.min(memberProfiles.size(), 2))) : memberProfiles;
        this.isPassword = team.getPassword() != null;
    }
}
