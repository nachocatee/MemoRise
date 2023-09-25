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

    public TeamListResponse(Long teamSeq, String teamName, Long ownerSeq, Long userSeq, String myProfile, String ownerProfile, List<String> memberProfiles, boolean isParticipated) {
        this.teamSeq = teamSeq;
        this.teamName = teamName;
        this.myProfile = userSeq != null ? myProfile : null;
        this.isParticipated = isParticipated;
        this.isOwner = ownerSeq == userSeq;
        this.ownerProfile = isOwner ? null : ownerProfile;
        this.memberProfiles = isParticipated ? (isOwner ? memberProfiles.subList(0, 3) : memberProfiles.subList(0, 2)) : memberProfiles;
    }
}
