package com.tjjhtjh.memorise.domain.tag.service;

import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.repository.MemoRepository;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.TaggedTeamRepository;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import com.tjjhtjh.memorise.domain.tag.service.dto.request.TaggedTeamRequest;
import com.tjjhtjh.memorise.domain.team.exception.NoTeamException;
import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TaggedTeamService {

    private final TaggedTeamRepository taggedTeamRepository;
    private final MemoRepository memoRepository;
    private final TeamRepository teamRepository;
    private static final String NO_MEMO_EXCEPTION = "해당하는 메모가 존재하지 않습니다";
    private static final String NO_TEAM_EXCEPTION = "해당하는 그룹이 존재하지 않습니다";


    @Transactional
    public void addTeamTag(TaggedTeamRequest taggedTeamRequest, Long memoId) throws MemoException {
        List<Long> addList = taggedTeamRequest.getTeamTaggedList();
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO_EXCEPTION));
        for (Long teamSeq : addList) {
            Team team = teamRepository.findByTeamSeqAndIsDeletedFalse(teamSeq)
                    .orElseThrow(()-> new NoTeamException(NO_TEAM_EXCEPTION));
            taggedTeamRepository.save(taggedTeamRequest.saveTeamToEntity(memo,team));
        }
    }

    @Transactional
    public void deleteTeamTag(Long memoId) throws MemoException {
        List<TaggedTeam> deleteTagList = taggedTeamRepository.findByTaggedTeamWithMemoSeq(memoId);
        for (TaggedTeam team : deleteTagList) {
            taggedTeamRepository.delete(team);
        }
    }

}
