package com.tjjhtjh.memorise.domain.tag.repository;

import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedTeamResponse;

import java.util.List;

public interface TaggedTeamRepositoryCustom {
    List<TaggedTeam> findByTaggedTeamWithMemoSeq(Long memoSeq);
    List<TaggedTeamResponse> findByTaggedTeamList(Long memoSeq);
}
