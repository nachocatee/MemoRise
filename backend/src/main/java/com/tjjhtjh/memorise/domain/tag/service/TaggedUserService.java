package com.tjjhtjh.memorise.domain.tag.service;

import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.memo.repository.MemoRepository;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.TaggedUserRepository;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
import com.tjjhtjh.memorise.domain.tag.service.dto.request.TaggedUserRequest;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.swing.text.html.HTML;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TaggedUserService {

    private final TaggedUserRepository taggedUserRepository;
    private final MemoRepository memoRepository;
    private final UserRepository userRepository;

    private static final String NO_MEMO_EXCEPTIOM = "해당하는 메모가 존재하지 않습니다";
    private static final String NO_USER_EXCEPTIOM = "해당하는 메모가 존재하지 않습니다";

    @Transactional
    public void addTagUser(TaggedUserRequest taggedUserRequest, Long memoId) throws MemoException {
        List<Long> addList = taggedUserRequest.getUserSeqList();
        Memo memo = memoRepository.findById(memoId)
                .orElseThrow(() -> new MemoException(NO_MEMO_EXCEPTIOM));
        for (Long userSeq : addList) {
            User user = userRepository.findByUserSeqAndIsDeletedFalse(userSeq)
                            .orElseThrow(()-> new NoUserException(NO_USER_EXCEPTIOM));
            taggedUserRepository.save(taggedUserRequest.saveUserToEntity(memo,user));
        }
    }

    @Transactional
    public void deleteTag(Long memoId) throws MemoException {
        List<TaggedUser> deleteTagList = taggedUserRepository.findByTaggedUserWithMemoSeq(memoId);
        for (TaggedUser user : deleteTagList) {
            taggedUserRepository.delete(user);
        }
    }
}
