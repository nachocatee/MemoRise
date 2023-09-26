package com.tjjhtjh.memorise.domain.tag.service.dto.request;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class TaggedTeamRequest {

    private Long teamSeq;
    private List<Long> teamTaggedList;

    public TaggedTeam saveTeamToEntity(Memo tagMemo, Team tagTeam){
        return TaggedTeam.builder()
                .team(tagTeam)
                .memo(tagMemo)
                .build();
    }

}
